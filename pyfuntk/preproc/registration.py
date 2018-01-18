##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import json
import numpy
import nibabel

# Package import
from pyfuntk.stats.utils import recursive_gzip
from pyfuntk.stats.utils import ungzip_file
from pyfuntk import DEFAULT_SPM_STANDALONE_PATH

# Nipype import
from nipype.interfaces import spm

# Pyconnectome import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.utils.segtools import bet2


def spm_normalize(datafile, tpm_file, outdir, with_segmentation=False,
                  extract_brain=False, extract_brain_thr=0.5, extra_files=None,
                  spmbin=DEFAULT_SPM_STANDALONE_PATH,
                  fslconfig=DEFAULT_FSL_PATH):
    """ SPM New Segment.

    Parameters
    ----------
    datafile: str
        the subject scan.
    tpm_file: str
        path to the tpm file.
    outdir: str
        the destination folder.
    with_segmentation: bool (optional, default False)
        if True use the newsegment normalization with dartel outputs.
    extract_brain: bool (optional, default False)
        if True use the BET2 routine to segment the subject brain.
    extract_brain_thr: float (optional, defaul 0.5)
        the BET2 fractional intensity threshold (0->1).
    extra_files: list of str (optional, default None)
        list of files to apply transform to.
    spmbin: str (optional, default DEFAULT_SPM_STANDALONE_PATH)
        the SPM standalone file.

    Returns
    -------
    deformation_file: str
        the estimated deformation field.
    normalized_file: str
        the registered input scan.
    dartel_files: list of str
        a list of files that can be used as dartel inputs.
    """
    # Configuration
    spm.SPMCommand.set_mlab_paths(
        matlab_cmd="{0} script ".format(spmbin), use_mcr=True)
    cwd = os.getcwd()
    os.chdir(outdir)

    # Extract/get brain if requested
    if extract_brain:
        (output, mask_file, mesh_file, outline_file, inskull_mask_file,
         inskull_mesh_file, outskull_mask_file, outskull_mesh_file,
         outskin_mask_file, outskin_mesh_file, skull_mask_file) = bet2(
            input_file=datafile,
            output_fileroot=outdir,
            outline=False,
            mask=True,
            skull=False,
            nooutput=False,
            f=extract_brain_thr,
            g=0,
            radius=None,
            smooth=None,
            c=None,
            threshold=False,
            mesh=False,
            shfile=fslconfig)
        datafile = ungzip_file(
            fname=output,
            prefix="",
            outdir=outdir)

    # Start normalization
    generated_files_to_compress = []
    dartel_files = []
    kwargs = {"apply_to_files": extra_files or []}
    if with_segmentation:

        # Define interface options
        tpm_struct = spm_tissue_probability_maps(tpm_file)

        # Generate the subject segmentation
        interface = segment = spm.NewSegment(
            channel_files=datafile,
            channel_info=(0.0001, 60, (True, True)),
            affine_regularization="mni",
            ignore_exception=False,
            tissues=tpm_struct,
            warping_regularization=4,
            write_deformation_fields=[True, True])
        runtime = interface.run()
        segmentation_outputs = runtime.outputs.get()
        for key in ["bias_corrected_images", "bias_field_images",
                    "dartel_input_images", "forward_deformation_field",
                    "inverse_deformation_field", "modulated_class_images",
                    "native_class_images", "normalized_class_images"]:
            generated_files_to_compress.append(segmentation_outputs[key])
        deformation_file = (
            segmentation_outputs["forward_deformation_field"] + ".gz")           
        dartel_files = segmentation_outputs["dartel_input_images"]

        # Apply the deformation to the subject scan
        kwargs["apply_to_files"].append(
            segmentation_outputs["bias_corrected_images"])
        interface = spm.Normalize12(
            jobtype="write",
            deformation_file=segmentation_outputs["forward_deformation_field"],
            write_bounding_box=[[-90, -126, -72], [91, 91, 109]],
            write_interp=4,
            write_voxel_sizes=[1, 1, 1],
            ignore_exception=False,
            **kwargs)
        runtime = interface.run()
        normalization_outputs = runtime.outputs.get()
        if isinstance(normalization_outputs["normalized_files"], list):
            normalized_file = normalization_outputs[
                "normalized_files"][-1] + ".gz"
            normalized_extra_files = [
                path + ".gz" 
                for path in normalization_outputs["normalized_files"][:-1]]
        else:
            normalized_file = normalization_outputs["normalized_files"] + ".gz"
            normalized_extra_files = []

        # Remove NANs
        im = nibabel.load(normalized_file.replace(".gz", ""))
        data = im.get_data()
        data[numpy.isnan(data)] = 0
        im = nibabel.Nifti1Image(data, affine=im.get_affine())
        nibabel.save(im, normalized_file)
        os.remove(normalized_file.replace(".gz", ""))

    else:

        # Normlization to template
        interface = spm.Normalize12(
            jobtype="estwrite",
            image_to_align=datafile,
            affine_regularization_type="mni",
            tpm=tpm_file,
            write_bounding_box=[[-90, -126, -72], [91, 91, 109]],
            write_interp=4,
            write_voxel_sizes=[1, 1, 1],
            ignore_exception=False,
            **kwargs)
        runtime = interface.run()
        normalization_outputs = runtime.outputs.get()
        for key in ["deformation_field", "normalized_image"]:
            generated_files_to_compress.append(normalization_outputs[key])
        deformation_file = normalization_outputs["deformation_field"] + ".gz"
        normalized_file = normalization_outputs["normalized_image"] + ".gz"
        if isinstance(normalization_outputs["normalized_files"], list):
            normalized_extra_files = [
                path + ".gz"
                for path in normalization_outputs["normalized_files"]]
        else:
            normalized_extra_files = None

    # GZip generated files
    output_files = recursive_gzip(obj=generated_files_to_compress)

    # Remove extra mfile and exit
    os.chdir(cwd)
    mfile = os.path.join(outdir, "pyscript.m")
    if os.path.isfile(mfile):
        os.remove(mfile)

    return (deformation_file, normalized_file, normalized_extra_files,
            dartel_files)


def spm_tissue_probability_maps(tpm_file):
    """ SPM tissue probability maps.

    Parameters
    ----------
    tpm_file: str
        path to the tpm file.

    Returns
    -------
    tpm_struct: list
        the spm tissue probability map description.
    """
    tissue1 = ((tpm_file, 1), 2, (True, True), (False, True))
    tissue2 = ((tpm_file, 2), 2, (True, True), (False, True))
    tissue3 = ((tpm_file, 3), 2, (True, True), (False, True))
    tissue4 = ((tpm_file, 4), 3, (False, False), (False, False))
    tissue5 = ((tpm_file, 5), 4, (False, False), (False, False))
    tpm_struct = [tissue1, tissue2, tissue3, tissue4, tissue5]
    return tpm_struct



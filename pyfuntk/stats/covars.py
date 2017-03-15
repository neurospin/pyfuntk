##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import nibabel
import numpy
import shutil

# PyFreeSurfer import
from pyfreesurfer.conversions.volconvs import mri_binarize
from pyfreesurfer.conversions.volconvs import mri_convert
from pyfreesurfer import DEFAULT_FREESURFER_PATH
from pyfreesurfer.wrapper import FSWrapper

# Package import
from pyfuntk import DEFAULT_SPM_STANDALONE_PATH
from .utils import erode_mask
from .utils import recursive_gzip

# Nipype import
from nipype.interfaces import spm


def extra_covars(subjfsdir, outdir, funcfile, deformationfile, iterations=1,
                 mask_label="wm", min_nb_of_voxels=50, nb_covars=3, verbose=0,
                 fsconfig=DEFAULT_FREESURFER_PATH,
                 spmbin=DEFAULT_SPM_STANDALONE_PATH):
    """ Erode a mask, perform a SVD, and select principle components to
    generate some non-interest covariates.

    Parameters
    ----------
    subjfsdir: str
        the subject FreeSurfer home directory.
    outdir: str
        the destination folder.
    funcfile: str
        a functional volume.
    deformationfile: str
        SPM file y_*.nii containing 3 deformation fields for the deformation in
        x, y and z dimension.
    iterations: int (optional, default 1)
        the number of path for the erosion.
    mask_label: str (optional, default 'wm')
        the name of the mask: 'wm' or 'ventricles'.
    min_nb_of_voxels: int (optional, default 50)
        a minimum number of voxels to perform properly the SVD.
    nb_covars: int (optional, default 3)
        the number of principle components to be extracted.
    verbose: int (optional, default 0)
        the verbosity level.
    fsconfig: str (optional, default DEFAULT_FREESURFER_PATH)
        the FreeSurfer configuration file.
    spmbin: str (optional, default DEFAULT_SPM_STANDALONE_PATH)
        the SPM standalone file.

    Returns
    -------
    native_maskfile: str
        the requested mask in the native space.
    erode_maskfile: str
        the eroded mask file.
    resample_maskfile: str
        the eroded mask file in the functional space.
    covars_file: str
        a text file with the requested number of principle components.
    """
    # Generate a mask from FreeSurfer
    if mask_label not in ("wm", "ventricles"):
        raise ValueError("Unexpected label '{0}'.".format(mask_label))
    asegfile = os.path.join(subjfsdir, "mri", "aseg.mgz")
    maskfile = os.path.join(outdir, "{0}.mgz".format(mask_label))
    kwargs = {mask_label: True}
    mri_binarize(
        inputfile=asegfile,
        outputfile=maskfile,
        match=None,
        inv=False,
        fsconfig=fsconfig,
        **kwargs)

    # Go back to native space
    reference_file = os.path.join(subjfsdir, "mri", "rawavg.mgz")
    native_maskfile = maskfile.replace(".mgz", ".nii")
    cmd = ["mri_convert", "--resample_type", "nearest", "--reslice_like",
           reference_file, maskfile, native_maskfile]
    process = FSWrapper(cmd, shfile=fsconfig)
    process()
    os.remove(maskfile)

    # Erode mask
    erode_maskfile = erode_mask(
        mask_file=native_maskfile,
        outdir=outdir,
        iterations=iterations)

    # Warp mask file to functional space
    spm.SPMCommand.set_mlab_paths(
        matlab_cmd="{0} script ".format(spmbin), use_mcr=True)
    cwd = os.getcwd()
    os.chdir(outdir)
    interface = spm.preprocess.Normalize12(
        deformation_file=deformationfile,
        jobtype="write",
        write_bounding_box=[[-78.0, -112.0, -50.0], [78.0, 76.0, 85.0]],
        write_interp=0,
        write_voxel_sizes=[3.0, 3.0, 3.0],
        apply_to_files=[erode_maskfile])
    runtime = interface.run()
    os.chdir(cwd)
    mfile = os.path.join(outdir, "pyscript.m")
    if os.path.isfile(mfile):
        os.remove(mfile)
    mfile = os.path.join(outdir, "pyscript_normalize12.m")
    if os.path.isfile(mfile):
        shutil.move(mfile,
                    os.path.join(outdir, "pyscript_normalize12_{0}.m".format(
                        mask_label)))

    # Extract covars from previous masks with at least N < min_nb_of_voxels
    resample_maskfile = runtime.outputs.get()["normalized_files"]
    mask_image = nibabel.load(resample_maskfile)
    mask_array = mask_image.get_data()
    mask_array[numpy.where(mask_array != 0)] = 1
    nb_voxels_in_mask = numpy.where(mask_array == 1)[0]
    if len(nb_voxels_in_mask) < min_nb_of_voxels:
        raise ValueError(
            "Not enough voxels ({0} < {1}) in mask '{2}'.".format(
                nb_voxels_in_mask, min_nb_of_voxels, resample_maskfile))

    # Compute a SVD
    func_array = nibabel.load(funcfile).get_data()
    timeseries = func_array[numpy.where(mask_array == 1)].T
    timeseries = timeseries.astype(float)
    timeseries -= timeseries.mean(axis=0)
    u, s, v = numpy.linalg.svd(timeseries, full_matrices=False)
    if verbose > 2:
        import matplotlib.pyplot as plt
        plt.plot(s)
        plt.show()

    # Get the covariates that represent the variability within the mask
    covars = u[:, :nb_covars]
    covars_file = os.path.join(outdir, "{0}.txt".format(mask_label))
    numpy.savetxt(covars_file, covars)

    # GZip generated files
    output_files = recursive_gzip(
        obj=[native_maskfile, erode_maskfile, resample_maskfile])
    output_files.append(covars_file)

    return output_files


def update_rpfile(rpile, covars_files, outdir, add_extra_mvt_reg=False):
    """ Complete a mouvment rp file with extra files containing covariates.

    Parameters
    ----------    
    rpile: str
        the mouvment rp file
    covars_files: list of str
        a list of extra files containing covariates.
    outdir: str
        the destination folder.
    add_extra_mvt_reg: bool (optional, default False)
        if set add the high order movement parameters: t-1, t+1.

    Returns
    -------
    completed_rpfile: str
        the completed mouvment rp file.   
    """
    # Load rp parameters
    parameters = numpy.loadtxt(rpile)

    # Add higher order mouvment parameters
    if add_extra_mvt_reg:

        # Add translation minus, plus shift
        # This in accordance to model2 of Imagen : NO ^2 and NO ^3
        t = parameters[:, :3]
        tplus = numpy.vstack((t[0, :], t[:-1, :]))
        tminus = numpy.vstack((t[1:, :], t[-1, :]))
        parameters = numpy.column_stack((parameters, tminus, tplus))

    # Go through each covars
    for path in covars_files:

        # Load the covariates and update parameters
        covars = numpy.loadtxt(path)
        parameters = numpy.column_stack((parameters, covars))

    # Save result
    completed_rpfile = os.path.join(outdir, os.path.basename(rpile))
    numpy.savetxt(completed_rpfile, parameters, fmt="%5.8f")
    
    return completed_rpfile

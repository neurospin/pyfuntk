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
import scipy.io
import nibabel
import matplotlib.pyplot as plt

# Package import
from .utils import get_onsets
from .utils import recursive_gzip
from pyfuntk import DEFAULT_SPM_STANDALONE_PATH

# Nipype import
from nipype.interfaces.base import Bunch
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces import spm


def spm_first_level(session_info, outdir, repetition_time, contrasts,
                    mask_image=None, mask_threshold=None,
                    spmbin=DEFAULT_SPM_STANDALONE_PATH):
    """ SPM fMRI First Level Analysis.

    Parameters
    ----------
    session_info: dict
        session info to leverage the first level design.
    outdir: str
        the destination folder.
    repetition_time: float
        the session repetition time.
    contrasts : list
        a list of contrasts with each contrast being a tuple of the form:
        ('name', 'stat', [condition list], [weight list]).
    mask_image: str (optional, default None)
        image for explicitly masking the analysis.
    mask_threshold: float (optional, default None)
        thresholding for the mask.
    spmbin: str (optional, default DEFAULT_SPM_STANDALONE_PATH)
        the SPM standalone file.

    Returns
    -------
    RPVimage: str
        resels per voxel image.
    beta_images: list of str
        design parameter estimates.
    mask_image: str
        binary mask to constrain estimation.
    residual_image: str
        mean-squared image of the residuals.
    spm_mat_file: str
        updated SPM mat file.
    design_snap: str
        a snapshot of the design matrix.
    con_files: list of str
        rename/converted contrast images from a t-contrast.
    spmT_files: list of str
        rename/converted stat images from a t-contrast.
    ess_files: list of str
        rename/converted contrast images from an F-contrast.
    spmF_files: list of str
        rename/converted stat images from an F-contrast.
    """
    # Configuration
    spm.SPMCommand.set_mlab_paths(
        matlab_cmd="{0} script ".format(spmbin), use_mcr=True)
    cwd = os.getcwd()
    os.chdir(outdir)

    # Generate the design matrix
    kwargs = {}
    if mask_image is not None:
        kwargs["mask_image"] = mask_image
    if mask_threshold is not None:
        kwargs["mask_threshold"] = mask_threshold
    interface = spm.Level1Design(
        session_info=session_info,
        timing_units="secs",
        microtime_resolution=16,
        microtime_onset=1,
        volterra_expansion_order=1,
        global_intensity_normalization="none",
        model_serial_correlations="AR(1)",
        factor_info=[{"name": {}}, {"levels": {}}],
        interscan_interval=repetition_time,
        bases={"hrf": {"derivs": [0, 0]}},
        **kwargs)
    runtime = interface.run()
    design_outputs = runtime.outputs.get()
    design_snap = spm_save_design(
        mat_file=design_outputs["spm_mat_file"],
        outdir=outdir)

    # Estimate the parameters of the model
    interface = spm.EstimateModel(
        estimation_method={"Classical": 1},
        spm_mat_file=design_outputs["spm_mat_file"])
    runtime = interface.run()
    estimate_outputs = runtime.outputs.get()

    # Estimate contrasts of interest
    interface = spm.EstimateContrast(
        spm_mat_file=estimate_outputs["spm_mat_file"],
        contrasts=contrasts,
        residual_image=estimate_outputs["residual_image"],
        beta_images=estimate_outputs["beta_images"])
    runtime = interface.run()
    contrast_outputs = runtime.outputs.get()
    os.chdir(cwd)

    # GZip generated files
    output_files = recursive_gzip(
        obj=[estimate_outputs["RPVimage"], estimate_outputs["beta_images"],
             estimate_outputs["mask_image"], estimate_outputs["residual_image"],
             estimate_outputs["spm_mat_file"]])

    # Encode/convert stat results
    con_images, spmT_images, ess_images, spmF_images = spm_encoding(
        con_images=contrast_outputs["con_images"],
        spmT_images=contrast_outputs["spmT_images"],
        ess_images=contrast_outputs["ess_images"],
        spmF_images=contrast_outputs["spmF_images"],
        contrasts=contrasts,
        outdir=outdir)
    output_files += [design_snap, con_images, spmT_images, ess_images,
                     spmF_images]

    # Remove extra mfile
    mfile = os.path.join(outdir, "pyscript.m")
    if os.path.isfile(mfile):
        os.remove(mfile)

    return output_files


def spm_save_design(mat_file, outdir, labels_size=4):
    """ Create a snap with the design matrix values.

    Parameters
    ----------
    mat_file: str
        the SPM mat file containing the desing matrix.
    outdir: str
        the destination folder.
    labels_size: int, default 4
        the label font size.

    Results
    -------
    design_snap: str
        a snapshot of the design matrix.   
    """
    # Get the design matrix
    spmmat = scipy.io.loadmat(mat_file, struct_as_record=False)
    design = spmmat["SPM"][0][0].xX[0][0]
    designmatrix = design.X
    designnames = []
    for item in design.name[0]:
        designnames.append(item[0])

    # Create a snapshot of the design matrix
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(designmatrix, cmap=plt.cm.Reds)
    ax.set_xticks(numpy.arange(0.5, designmatrix.shape[1]))
    ax.tick_params(which="both", axis="both", width=0, length=0)
    ax.set_xticklabels(designnames, size=labels_size, rotation=90)
    colorbar = fig.colorbar(heatmap)
    fig.tight_layout()

    # Save to PNG file
    design_snap = os.path.join(outdir, "spm_design.png")
    plt.savefig(design_snap)

    # Release memory
    fig.clear()
    plt.close()

    return design_snap


def spm_encoding(con_images, spmT_images, ess_images, spmF_images,
                 contrasts, outdir):
    """ Encode the SPM statistics in compress nifti format and rename files
    to include the contrast name.

    Parameters
    ----------
    con_images: list of str
        contrast images from a t-contrast.
    spmT_images: list of str
        stat images from a t-contrast.
    ess_images: list of str
        contrast images from an F-contrast.
    spmF_images: list of str
        stat images from an F-contrast.
    contrasts : list
        a list of contrasts with each contrast being a tuple of the form:
        ('name', 'stat', [condition list], [weight list]).
    outdir: str
        the destination folder.

    Returns
    -------
    con_nii_images: list of str
        rename/converted contrast images from a t-contrast.
    spmT_nii_images: list of str
        rename/converted stat images from a t-contrast.
    ess_nii_images: list of str
        rename/converted contrast images from an F-contrast.
    spmF_nii_images: list of str
        rename/converted stat images from an F-contrast.   
    """
    con_nii_images = []
    spmT_nii_images = []
    ess_nii_images = []
    spmF_nii_images = []
    for images, nii_images in [(con_images, con_nii_images),
                               (spmT_images, spmT_nii_images),
                               (ess_images, ess_nii_images),
                               (spmF_images, spmF_nii_images)]:
        if isinstance(images, basestring) and os.path.isfile(images):
            images = [images]            
        if isinstance(images, list):
            for path in images:

                # Build the image file name
                name = os.path.basename(path).split(".")[0]
                index = int(name.split("_")[1]) - 1
                contrast_name = contrasts[index][0].lower().replace(" ", "_")
                basename = "{0}_{1}.nii.gz".format(name, contrast_name)
                outfile = os.path.join(outdir, basename)

                # Use nibabel for the image format conversion
                image = nibabel.load(path)
                nibabel.save(image, outfile)
                os.remove(path)

                # Save destination path
                nii_images.append(outfile)

    return con_nii_images, spmT_nii_images, ess_nii_images, spmF_nii_images


def spm_model_specification(behavioral_data, fmri_sessions, onset_name,
                            condition_name, duration_name, time_repetition,
                            realignment_parameters, delimiter, start,
                            outdir, concatenate_runs=True,
                            high_pass_filter_cutoff=128):
    """ Specify the SPM model used in the GLM and estimate the design matrix.

    .. note::

        * `fmri_sessions` and `behavioral_data` must have the same number
          of elements.
        * `onsets` and `durations` values must have the same units as the
          TR used in the processings (ie. seconds).

    Parameters
    ----------
    behavioral_data: list of str
        list of CSV session behavioral data.
    fmri_sessions: list of str
        list of path to fMRI sessions.
    onset_name: str
        the name of the column in the `behavioral_data` file containing the
        onsets.
    condition_name: str
        the name of the column in the `behavioral_data` file containing the
        conditions.
    duration_name: str
        the name of the column in the `behavioral_data` file containing the
        condition durations.
    time_repetition: float
        the repetition time in seconds (in seconds).
    realignment_parameters": str
        path to the SPM realign output parameters.
    delimiter: str
        separator used to split the `behavioral_data` file.
    start: int
        line from which we start reading the `behavioral_data` file.
    outdir: str
        Where to store the output file.
    concatenate_runs: bool (optional, default True)
        concatenate all runs to look like a single session.
    high_pass_filter_cutoff: float (optional, default 128)
        high-pass filter cutoff in secs.

    Returns
    -------
    session_info: dict
        session info to leverage the first level design.
    model_specifications: str
        file containing all model specifications.   
    """
    # Assert that we have one behavioral data per session
    if len(behavioral_data) != len(fmri_sessions):
        raise ValueError("One behavioral data per session is required, "
                         "got {0} behaviral data and {1} session.".format(
                             len(behavioral_data), len(fmri_sessions)))

    # Get each session acquisition conditions
    info = []
    for csvfile in behavioral_data:

        # Parse the behavioural file
        all_onsets = get_onsets(csvfile,
                                condition_name, onset_name, duration_name,
                                delimiter, start)

        # Create a nipype Bunch (dictionary-like) structure
        conditions = []
        onsets = []
        durations = []
        for condition_name, item in all_onsets.items():
            conditions.append(condition_name)
            if str(item["onsets"][0]) == "nan":
                onsets.append([numpy.nan])
                durations.append([numpy.nan])
            else:
                onsets.append([float(x) for x in item["onsets"]])
                durations.append([float(x) for x in item["durations"]])
        info.append(
            Bunch(conditions=conditions, onsets=onsets, durations=durations))

    # Make a model specification compatible with SPM designer
    spec_interface = SpecifySPMModel(
        concatenate_runs=concatenate_runs,
        input_units="secs",
        output_units="secs",
        time_repetition=time_repetition,
        high_pass_filter_cutoff=high_pass_filter_cutoff,
        functional_runs=fmri_sessions,
        subject_info=info,
        realignment_parameters=realignment_parameters)
    spec_interface.run()

    # The previous interface use numpy in double precision. In order to be
    # python-json compliant need to cast expicitely all float items
    def cast_to_float(obj):
        """ Recursive method that cast numpy.double items.

        Parameters
        ----------
        obj: object
            a generic python object.

        Returns
        -------
        out: object
            the float-casted input object.
        """
        # Deal with dictionary
        if isinstance(obj, dict):
            out = {}
            for key, val in obj.items():
                out[key] = cast_to_float(val)

        # Deal with tuple and list
        elif isinstance(obj, (list, tuple)):
            out = []
            for val in obj:
                out.append(cast_to_float(val))
            if isinstance(obj, tuple):
                out = tuple(out)

        # Otherwise cast if it is a numpy.double
        else:
            out = obj
            if isinstance(obj, float):
                out = float(obj)

        return out

    # Save the design parameters
    session_info = cast_to_float(spec_interface.aggregate_outputs().get()[
        "session_info"])
    model_specifications = os.path.join(outdir, "model_specifications.json")
    with open(model_specifications, "w") as _file:
        json.dump(session_info, _file, indent=4)

    return session_info, model_specifications

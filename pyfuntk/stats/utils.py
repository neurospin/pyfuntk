##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import csv
import gzip
import numpy
from collections import defaultdict
from scipy.ndimage.morphology import binary_erosion
import nibabel



def get_onsets(csvfile, condition_name, onset_name, duration_name,
               delimiter="\t", start=0):
    """ Load a CSV behavioral data.
    """

    # Read the csv
    onsets = defaultdict(list)
    with open(csvfile, "rb") as ocsv:
        # > move the pointer to the header
        for i in range(start):
            next(ocsv)
        reader = csv.DictReader(ocsv, delimiter=delimiter, quotechar='|')
        for row in reader:
            for (k, v) in row.items():
                onsets[k].append(v)

    # Create conditions
    conditions = { x : {}
                  for x in set([x for x in onsets[condition_name] if x!=""]) }
    for condition in conditions.keys():
        indices = [i for i, x in enumerate(onsets[condition_name])
                   if x == condition]
        conditions[condition]["onsets"] = [onsets[onset_name][i]
                                           for i in indices]
        conditions[condition]["durations"] = [onsets[duration_name][i]
                                              for i in indices]

    return conditions


def erode_mask(mask_file, outdir, iterations=1):
    """ Erode a binary mask file.

    Parameters
    ----------
    mask_file: str (mandatory)
        the mask to erode.
    outdir: str (mandatory)
        the folder where the output results will be saved.
    iterations: int (optional, default 1)
        the number of path for the erosion.

    Returns
    -------
    erode_file: str
        the eroded binary mask file.
    """
    # Generate structural element
    structuring_element = numpy.array(
        [[[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]],
         [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]],
         [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]])

    # Erode source mask
    source_image = nibabel.load(mask_file)
    source_data = source_image.get_data()
    erode_data = binary_erosion(source_data, iterations=iterations,
                                structure=structuring_element)
    erode_data = erode_data.astype(source_data.dtype)
    erode_image = nibabel.Nifti1Image(erode_data, source_image.get_affine())
    erode_file = os.path.join(
        outdir, "e{0}".format(os.path.basename(mask_file)))
    nibabel.save(erode_image, erode_file)

    return erode_file


def normalize_array(filepath, outdir):
    """ Get a numpy array from a file and normalize the data column by column.

    Parameters
    ----------
    filepath: str
        an input file containing the numpy array.
    outdir: str
        the destination folder.

    Returns
    -------
    outfile: str
        the output file with normalized data array.
    """
    array = numpy.loadtxt(filepath)
    array = (array - array.mean(axis=0)) / array.std(axis=0)
    outfile = os.path.join(outdir, os.path.basename(filepath))
    numpy.savetxt(outfile, array, fmt="%5.8f")

    return outfile


def ungzip_file(fname, prefix="u", outdir=None):
    """ Copy and ungzip the input file.

    
    Parameters
    ----------
    fname: list of str
        input file to ungzip.
    prefix: str (optional, default 'u')
        the prefix of the result file.
    outdir: str (optional, default None)
        the output directory where ungzip file is saved. If not set use the
        input image directory.

    Returns
    -------
    ungzipfile: str
        the ungzip file.
    """
    # Check the input file exists on the file system
    if not os.path.isfile(fname):
        raise ValueError("'{0}' is not a valid filename.".format(fname))

    # Check that the outdir is valid
    if outdir is not None:
        if not os.path.isdir(outdir):
            raise ValueError(
                "'{0}' is not a valid directory.".format(outdir))
    else:
        outdir = os.path.dirname(fname)

    # Get the file descriptors
    base, extension = os.path.splitext(fname)
    basename = os.path.basename(base)

    # Ungzip only known extension
    if extension in [".gz"]:

        # Generate the output file name
        basename = prefix + basename
        ungzipfile = os.path.join(outdir, basename)

        # Read the input gzip file
        with gzip.open(fname, "rb") as gzfobj:
            data = gzfobj.read()

        # Write the output ungzip file
        with open(ungzipfile, "wb") as openfile:
            openfile.write(data)

    # Default, unknown compression extension: the input file is returned
    else:
        ungzipfile = fname

    return ungzipfile


def ungzip_list_of_files(files, prefix="u", outdir=None):
    """ Copy and ungzip the input files.
    
    Parameters
    ----------
    files: list of str
        input files to ungzip.
    prefix: str (optional, default 'u')
        the prefix of the result file.
    outdir: str (optional, default None)
        the output directory where ungzip file is saved. If not set use the
        input image directory.

    Returns
    -------
    ungzipfiles: list of str
        the ungzip files.   
    """
    ungzipfiles = []
    for fname in files:
        ungzipfiles.append(ungzip_file(fname, prefix, outdir))

    return ungzipfiles


def gzip_file(ungzip_file, prefix="g", outdir=None,
              remove_original_file=False):
    """ Gzip an input file and possibly remove the original file.

    Parameters
    ----------
    ungzip_file: str (mandatory)
        an input file to gzip.
    prefix: str (optional, default 'g')
        a prefix that will be concatenated to the produced file basename.
    outdir: str (optional, default None)
        the destination folder where the Gzip file is saved. If this parameter
        is None, the input image folder is considered as an output folder.
    remove_original_file: bool (optiona, default False)
        if True, remove the original file.

    Returns
    -------
    gzip_file: str
        the returned Gzip file.
    """
    # Check the input file exists on the file system
    if not os.path.isfile(ungzip_file):
        raise ValueError("'{0}' is not a valid filename.".format(ungzip_file))

    # Check that the outdir is valid
    if outdir is not None:
        if not os.path.isdir(outdir):
            raise ValueError("'{0}' is not a valid directory.".format(outdir))
    else:
        outdir = os.path.dirname(ungzip_file)

    # Get the file descriptors
    dirname, basename = os.path.split(ungzip_file)
    base, extension = os.path.splitext(basename)

    # Gzip only non compressed file
    if extension not in [".gz", ".json", ".txt"]:

        # Generate the output file name
        basename = base + extension + ".gz"
        if prefix:
            basename = prefix + basename
        gzip_file = os.path.join(outdir, basename)

        # Write the output gzip file
        with open(ungzip_file, "rb") as openfile:
            with gzip.open(gzip_file, "w") as gzfobj:
                gzfobj.writelines(openfile)

        # Remove original file if requested
        if remove_original_file:
            os.remove(ungzip_file)

    # Default, the input file is returned
    else:
        gzip_file = ungzip_file

    return gzip_file


def recursive_gzip(obj):
    """ Recursively find and Gzip files.

    Parameters
    ----------
    obj: object (mandatory)
        a Python object containing files to Gzip.

    Returns
    -------
    gzip_obj: object
        the input object with Gziped files.
    """
    # Stop case: a non iterative structure is reached
    if isinstance(obj, basestring) and os.path.isfile(obj):
        return gzip_file(obj, prefix="", outdir=None,
                         remove_original_file=True)

    # Go deeper
    elif isinstance(obj, (tuple, list)):
        gzip_obj = []
        for item in obj:
            gzip_obj.append(recursive_gzip(item))
        if isinstance(obj, tuple):
            gzip_obj = tuple(gzip_obj)
        return gzip_obj

    # Default return object
    else:
        return obj

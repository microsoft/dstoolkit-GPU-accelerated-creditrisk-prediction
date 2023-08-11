import argparse
import os
import tarfile
import time
import urllib.request

from azureml.core import Run


def extract_files(fname: str, extraction_dir: str) -> None:
    """Extracts the contents of a tar.gz file to a directory.

    Args:
        fname (str): Path to the tar.gz file.
        extraction_dir (str): Path to the directory where the contents of the tar.gz file will be extracted.

    Returns:
        None
    """
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(extraction_dir)
    tar.close()
    print(f"Successfully extracted contents of '{fname}' to '{extraction_dir}'")
    print(f"Contents in directory '{extraction_dir}': {os.listdir(extraction_dir)}")

    print("Directory Structure:")
    print("--------------------------------")
    print()
    for root, dirs, files in os.walk(extraction_dir, topdown=True):
        print(root)
        print(dirs)
        print(files)
        print("--------------------------------")


def download_from_url(url: str, dir_: str) -> str:
    """Downloads a file from a URL to a directory.

    Args:
        url (str): URL of the file to download.
        dir_ (str): Path to the directory where the file will be downloaded.

    Returns:
        str: Path to the downloaded file.
    """
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fname = url.split("/")[-1]
    fname = os.path.join(dir_, fname)

    if not os.path.exists(fname):
        max_attempts = 80
        attempts = 0
        sleeptime = 10
        while attempts < max_attempts:
            time.sleep(sleeptime)
            try:
                response = urllib.request.urlopen(url, timeout=5)
                content = response.read()
                f = open(fname, "wb")
                f.write(content)
                f.close()
                print("Successfully downloaded data!")
                break
            except Exception as e:
                attempts += 1
                print(type(e))
                print(e)
    return fname


def download_and_extract_data(url: str, dir_: str) -> None:
    """Downloads a tar.gz file from a URL and extracts its contents to a directory.

    Args:
        url (str): URL of the tar.gz file to download.
        dir_ (str): Path to the directory where the tar.gz file will be downloaded and its contents extracted.

    Returns:
        None
    """
    fname = download_from_url(url, dir_)
    extract_files(fname, dir_)


def parse_args_() -> argparse.Namespace:
    """Parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-url",
        type=str,
        help="URL of the data to download",
        default="http://rapidsai-data.s3-website.us-east-2.amazonaws.com/notebook-mortgage-data/mortgage_2000-2016.tgz",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory to download the data to",
        default="data",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function.

    Returns:
        None
    """
    run = Run.get_context()
    ws = run.experiment.workspace
    args = parse_args_()
    print("Hello")
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    data_folder = os.path.join(cwd, args.data_dir)
    download_and_extract_data(args.data_url, data_folder)
    acq_file_folder = os.path.join(data_folder, "acq/")
    perf_file_folder = os.path.join(data_folder, "perf/")
    acq_files = [os.path.join(acq_file_folder, f) for f in os.listdir(acq_file_folder)]
    perf_files = [
        os.path.join(perf_file_folder, f) for f in os.listdir(perf_file_folder)
    ]
    datastore = ws.get_default_datastore()
    datastore.upload_files(
        acq_files,
        target_path="credit_risk_data/acq/",
        overwrite=True,
        show_progress=True,
    )
    datastore.upload_files(
        perf_files,
        target_path="credit_risk_data/perf/",
        overwrite=True,
        show_progress=True,
    )
    print("Uploaded files to datastore")
    run.complete()


if __name__ == "__main__":
    main()

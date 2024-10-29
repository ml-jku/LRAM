import argparse
import joblib
import subprocess
from pathlib import Path
from tqdm import tqdm
from joblib import delayed


class ProgressParallel(joblib.Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        

def copy_file(file_path, source, destination):
    relative_path = Path(file_path).relative_to(source)
    destination_path = Path(destination) / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cp', '-p', str(file_path), str(destination_path)], check=True)
    

def main(source, destination, n_jobs, suffix=None):
    print("Collecting files.")
    pattern = f"**/*.{suffix}" if suffix is not None else "*"
    file_list = [file for file in Path(source).rglob(pattern) if file.is_file()]
    print(f"Copying {len(file_list)} files.")
    ProgressParallel(n_jobs=n_jobs, total=len(file_list))(delayed(copy_file)(file, source, destination) for file in file_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel cp with progress bar.")
    parser.add_argument('--src', type=str, help='Source directory to copy from.')
    parser.add_argument('--dst', type=str, help='Destination directory to copy to.')
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs.')
    args = parser.parse_args()
    main(args.src, args.dst, args.n_jobs, args.suffix)
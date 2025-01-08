#!/usr/bin/env python3
import glob
import os
import re
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYTORCH_REPOS = {
    "pytorch": "https://github.com/pytorch/pytorch.git",
    "vision": "https://github.com/pytorch/vision.git",
    "audio": "https://github.com/pytorch/audio.git",
    "text": "https://github.com/pytorch/text.git"
}


def get_rocm_info() -> Tuple[str, str]:
    """Rileva l'installazione di ROCm e la sua versione."""
    default_path = "/opt/rocm"
    if not os.path.exists(default_path):
        raise RuntimeError(f"ROCm path {default_path} non trovato")

    try:
        result = subprocess.run(['rocm-smi', '--showversion'],
                                capture_output=True, text=True, check=True)
        version_match = re.search(r'ROCm-(\d+\.\d+\.\d+)', result.stdout)
        if version_match:
            return default_path, version_match.group(1)
    except subprocess.CalledProcessError:
        pass

    rocm_dirs = glob.glob("/opt/rocm-*")
    if rocm_dirs:
        latest_dir = sorted(rocm_dirs)[-1]
        version_match = re.search(r'rocm-(\d+\.\d+\.\d+)', latest_dir)
        if version_match:
            return default_path, version_match.group(1)

    raise RuntimeError("Non è stato possibile determinare la versione di ROCm")


def setup_build_env(rocm_path: str, gpu_arch: str) -> Dict[str, str]:
    """Configura l'ambiente di build per ROCm."""
    env = os.environ.copy()

    env.update({
        'ROCM_HOME': rocm_path,
        'ROCM_PATH': rocm_path,
        'HIP_PATH': f"{rocm_path}/hip",
        'HIP_PLATFORM': 'amd',
        'HIP_COMPILER': 'clang',
        'HIP_RUNTIME': 'rocclr',
        'HIP_CLANG_PATH': f"{rocm_path}/llvm/bin",
        'HIPCC_PATH': f"{rocm_path}/bin/hipcc",
        'PATH': f"{rocm_path}/bin:{rocm_path}/llvm/bin:{env.get('PATH', '')}",
        'LD_LIBRARY_PATH': f"{rocm_path}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    })

    hip_flags = [
        f'--rocm-path={rocm_path}',
        f'--offload-arch={gpu_arch}',
        '-D__HIP_PLATFORM_AMD__=1',
        '-D_GLIBCXX_USE_CXX11_ABI=0',
        '-Wno-missing-prototypes',
        '-Wno-error=missing-prototypes'
    ]

    env.update({
        'HIPCC_COMPILE_FLAGS_APPEND': ' '.join(hip_flags),
        'HIP_HIPCC_FLAGS': ' '.join(hip_flags),
        'CMAKE_HIP_FLAGS': ' '.join(hip_flags),
        'CMAKE_PREFIX_PATH': f"{rocm_path}/lib/cmake/hip:{env.get('CMAKE_PREFIX_PATH', '')}",
        'USE_ROCM': '1',
        'PYTORCH_ROCM_ARCH': gpu_arch,
        'USE_NINJA': '1',
        'BUILD_TEST': '0'
    })

    return env


def clone_repo(repo_name: str, target_path: Path) -> None:
    """Clona una repository PyTorch se non esiste."""
    if not target_path.exists():
        logger.info(f"Clonazione {repo_name} in {target_path}")
        subprocess.run(["git", "clone", "--recurse", PYTORCH_REPOS[repo_name], str(target_path)], check=True)

    setup_repo(target_path)


def setup_repo(repo_path: Path) -> None:
    """Configura una repository per la build."""
    logger.info(f"Configurazione {repo_path}")

    subprocess.run(["git", "submodule", "sync", "--recursive"], cwd=repo_path, check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_path, check=True)

    requirements = [
        "astunparse", "numpy", "pyyaml", "typing_extensions",
        "future", "six", "requests", "setuptools", "wheel",
        "cmake", "ninja"
    ]

    if (repo_path / "requirements.txt").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                       cwd=repo_path, check=True)

    subprocess.run([sys.executable, "-m", "pip", "install"] + requirements, check=True)


def init_repo(repo_path: Path) -> None:
    """Inizializza il repository con submodules."""
    logger.info("Inizializzazione submodules...")
    subprocess.run(["git", "submodule", "sync", "--recursive"], cwd=repo_path, check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_path, check=True)


def build_package(source_path: Path, python_cmd: str, gpu_arch: str) -> bool:
    """Compila un pacchetto PyTorch."""
    try:
        init_repo(source_path)
        rocm_path, rocm_version = get_rocm_info()
        logger.info(f"ROCm {rocm_version} trovato in {rocm_path}")

        env = setup_build_env(rocm_path, gpu_arch)
        os.chdir(source_path)

        build_cmd = [python_cmd, 'setup.py', 'bdist_wheel']
        logger.info(f"Avvio build di {source_path.name} con {' '.join(build_cmd)}")

        subprocess.run(build_cmd, env=env, check=True)

        wheels = list(Path('dist').glob('*.whl'))
        if wheels:
            logger.info(f"Build completata: {wheels[-1]}")
            return True

        logger.error("Nessun wheel trovato dopo la build")
        return False

    except Exception as e:
        logger.error(f"Errore durante la build di {source_path.name}: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Uso: script.py <base_path> [python_cmd] [gpu_arch] [packages...]")
        print("Packages disponibili:", ", ".join(PYTORCH_REPOS.keys()))
        #sys.exit(1)

    base_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("pytorch/")
    python_cmd = sys.argv[2] if len(sys.argv) > 2 else "python3.11"
    gpu_arch = sys.argv[3] if len(sys.argv) > 3 else "gfx1102"
    packages = sys.argv[4:] if len(sys.argv) > 4 else ["pytorch"]

    invalid_packages = set(packages) - set(PYTORCH_REPOS.keys())
    if invalid_packages:
        logger.error(f"Pacchetti non validi: {invalid_packages}")
        sys.exit(1)

    base_path.mkdir(parents=True, exist_ok=True)

    for package in packages:
        package_path = base_path / package
        clone_repo(package, package_path)
        if not build_package(package_path, python_cmd, gpu_arch):
            logger.error(f"Build di {package} fallita")
            sys.exit(1)

    logger.info("Build completata con successo")


if __name__ == '__main__':
    main()
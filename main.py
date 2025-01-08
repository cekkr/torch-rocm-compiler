#!/usr/bin/env python3
import glob
import os
import re
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_rocm_info():
    """Rileva l'installazione di ROCm e la sua versione."""
    default_path = "/opt/rocm"
    if not os.path.exists(default_path):
        raise RuntimeError(f"ROCm path {default_path} non trovato")

    # Cerca la versione installata
    try:
        result = subprocess.run(['rocm-smi', '--showversion'],
                                capture_output=True, text=True, check=True)
        version_match = re.search(r'ROCm-(\d+\.\d+\.\d+)', result.stdout)
        if version_match:
            return default_path, version_match.group(1)
    except subprocess.CalledProcessError:
        pass

    # Fallback: cerca nelle directory
    rocm_dirs = glob.glob("/opt/rocm-*")
    if rocm_dirs:
        latest_dir = sorted(rocm_dirs)[-1]
        version_match = re.search(r'rocm-(\d+\.\d+\.\d+)', latest_dir)
        if version_match:
            return default_path, version_match.group(1)

    raise RuntimeError("Non è stato possibile determinare la versione di ROCm")

def setup_build_env(rocm_path, gpu_arch):
    env = os.environ.copy()

    # Percorsi base
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

    # Flag di compilazione
    hip_flags = [
        f'--rocm-path={rocm_path}',
        f'--offload-arch={gpu_arch}',
        '-D__HIP_PLATFORM_AMD__=1',
        '-D_GLIBCXX_USE_CXX11_ABI=0'
    ]

    env['HIPCC_COMPILE_FLAGS_APPEND'] = ' '.join(hip_flags)
    env['HIP_HIPCC_FLAGS'] = ' '.join(hip_flags)
    env['CMAKE_HIP_FLAGS'] = ' '.join(hip_flags)
    env['CMAKE_PREFIX_PATH'] = f"{rocm_path}/lib/cmake/hip:{env.get('CMAKE_PREFIX_PATH', '')}"

    # Flag PyTorch specifici
    env['USE_ROCM'] = '1'
    env['PYTORCH_ROCM_ARCH'] = gpu_arch
    env['USE_NINJA'] = '1'

    return env


def build_pytorch(source_path, python_cmd, gpu_arch):
    try:
        rocm_path, rocm_version = get_rocm_info()
        logger.info(f"ROCm {rocm_version} trovato in {rocm_path}")

        env = setup_build_env(rocm_path, gpu_arch)
        os.chdir(source_path)

        build_cmd = [python_cmd, 'setup.py', 'bdist_wheel']
        logger.info(f"Avvio build con {' '.join(build_cmd)}")

        subprocess.run(build_cmd, env=env, check=True)

        wheels = list(Path('dist').glob('*.whl'))
        if wheels:
            logger.info(f"Build completata: {wheels[-1]}")
            return True

        logger.error("Nessun wheel trovato dopo la build")
        return False

    except Exception as e:
        logger.error(f"Errore durante la build: {e}")
        return False


def main():
    source_path = sys.argv[1] if len(sys.argv) > 1 else "/home/riccardo/Sources/Gits/pytorch"
    python_version = sys.argv[2] if len(sys.argv) > 2 else "python3.11"
    gpu_arch = sys.argv[3] if len(sys.argv) > 3 else "gfx1102"

    if not os.path.exists(source_path):
        logger.error(f"Directory sorgente non trovata: {source_path}")
        sys.exit(1)

    success = build_pytorch(source_path, python_version, gpu_arch)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
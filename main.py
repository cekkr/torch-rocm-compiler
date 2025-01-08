#!/usr/bin/env python3
import os
import sys
import subprocess
import glob
import re
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_rocm_installation():
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


def check_clang():
    """Verifica la presenza e la versione di clang++."""
    rocm_path = "/opt/rocm"
    clang_path = f"{rocm_path}/llvm/bin/clang++"

    if not os.path.exists(clang_path):
        raise RuntimeError(f"clang++ non trovato in {clang_path}")

    try:
        result = subprocess.run([clang_path, '--version'],
                                capture_output=True, text=True, check=True)
        logger.info(f"Trovato clang++: {result.stdout.splitlines()[0]}")
        return clang_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Errore nel verificare clang++: {e}")


def prepare_build_environment(rocm_path: str, rocm_version: str, gpu_arch: str):
    """Prepara l'ambiente di build con tutte le variabili necessarie."""
    major, minor, patch = rocm_version.split('.')

    env = os.environ.copy()
    env.update({
        'ROCM_PATH': rocm_path,
        'ROCM_HOME': rocm_path,
        'HIP_PATH': f"{rocm_path}/hip",
        'HIP_PLATFORM': 'amd',
        'HIP_COMPILER': 'clang',
        'PYTORCH_ROCM_ARCH': gpu_arch,
        'USE_ROCM': '1',
        'USE_CUDA': '0',
        'PATH': f"{rocm_path}/bin:{rocm_path}/llvm/bin:{env.get('PATH', '')}",
        'LD_LIBRARY_PATH': f"{rocm_path}/lib:{rocm_path}/lib64:{env.get('LD_LIBRARY_PATH', '')}",
    })

    # Flags di compilazione principali
    compile_flags = [
        '--rocm-path=/opt/rocm',  # Path assoluto senza quote
        '--rocm-device-lib-path=/opt/rocm/amdgcn/bitcode',
        f'--offload-arch={gpu_arch}',
        '-I/opt/rocm/include',
        '-I/opt/rocm/include/hip',
        '-I/opt/rocm/include/rocm',
        '-D__HIP_PLATFORM_AMD__',
        '-D__HIP_ROCclr__',
        f'-DROCM_VERSION_MAJOR={major}',
        f'-DROCM_VERSION_MINOR={minor}',
        f'-DROCM_VERSION={major}{minor}301',  # Format: 60301
        f'-DTORCH_HIP_VERSION={major}{minor}2',  # Format: 602
        '-DUSE_ROCM',
        '-DHIP_COMPILER=clang',
        '-DUSE_MIOPEN',
        '-DHIPBLAS_V2',
        '-DHIP_NEW_TYPE_ENUMS',
        '-std=c++17',
        '-fPIC',
        '-pthread',
    ]

    env['HIPCC_COMPILE_FLAGS_APPEND'] = ' '.join(compile_flags)

    # CMake arguments
    cmake_args = [
        # Base ROCm config
        f'-DROCM_PATH={rocm_path}',
        f'-DHIP_PATH={rocm_path}/hip',
        '-DHIP_COMPILER=clang',
        '-DUSE_ROCM=ON',
        '-DUSE_CUDA=OFF',
        '-DCMAKE_CXX_STANDARD=17',

        # HIP/ROCm specific flags
        f'-DCMAKE_HIP_COMPILE_FLAGS="--rocm-path={rocm_path}"',
        f'-DHIP_HIPCC_FLAGS="--rocm-path={rocm_path}"',
        f'-DHIP_CLANG_FLAGS="--rocm-path={rocm_path}"',
        f'-DCMAKE_HIP_LINK_FLAGS="--rocm-path={rocm_path}"',
        f'-DHIP_COMPILER_FLAGS="--rocm-path={rocm_path} --offload-arch={gpu_arch}"',
        f'-DCMAKE_HIP_ARCHITECTURES={gpu_arch}',
        f'-DHIP_CLANG_PATH={rocm_path}/llvm/bin',
        f'-DHIP_RUNTIME_PATH={rocm_path}/hip',

        # FBGEMM specific flags
        '-DCMAKE_BUILD_TYPE=Release',
        '-DUSE_NATIVE_ARCH=OFF',
        '-DFBGEMM_NO_AVX512=ON',
        '-DBUILD_CAFFE2=OFF',

        # Compiler flags
        '-DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -mno-avx2"',
        '-DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -mno-avx2"',

        # Debug flags
        '-DCMAKE_VERBOSE_MAKEFILE=ON',
        '-DCMAKE_HIP_VERBOSE_MAKEFILE=ON',
        '-DCMAKE_HIP_VERBOSE_COMPILATION=ON'
    ]

    cmake_args.extend([
        f'-DCMAKE_HIP_FLAGS="--rocm-path={rocm_path} ${{CMAKE_HIP_FLAGS}}"',
        f'-DHIP_HIPCC_FLAGS="--rocm-path={rocm_path} ${{HIP_HIPCC_FLAGS}}"',
        f'-DCMAKE_HIP_COMPILER_FLAGS="--rocm-path={rocm_path}"',
    ])

    env['CMAKE_ARGS'] = ' '.join(cmake_args)
    return env

def patch_gloo_cmake(source_path: str, rocm_path: str):
    """Patcha il file CMake di gloo per forzare rocm-path."""

    # todo: make it dynamic in basis of the system
    HIP_HIPCC_FLAGS = "--rocm-path=/opt/rocm \
                     --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode \
                     --offload-arch=gfx1102 \
                     -D__HIP_PLATFORM_AMD__ \
                     -D__HIP_ROCclr__ \
                     -DUSE_ROCM \
                     -DHIP_COMPILER=clang \
                     -std=c++17 \
                     -fPIC \
                     -pthread"

    gloo_cmake = os.path.join(source_path, "third_party/gloo/CMakeLists.txt")
    with open(gloo_cmake, 'r') as f:
        lines = f.readlines()

    # Cerca il punto di inserimento corretto
    for i, line in enumerate(lines):
        if 'set(CMAKE_HIP_FLAGS' in line:
            # Assicurati che il flag rocm-path sia presente
            if f'--rocm-path={rocm_path}' not in line:
                lines[i] = line.rstrip() + f' "--rocm-path={rocm_path}"\n'
        elif 'hip_add_executable' in line or 'hip_add_library' in line:
            # Aggiungi anche per i target HIP
            prev_line = lines[i-1]
            if not any('--rocm-path' in l for l in [prev_line, line]):
                lines.insert(i, f'set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS} "--rocm-path={rocm_path}")\n')

    with open(gloo_cmake, 'w') as f:
        f.writelines(lines)

def build_pytorch(source_path: str, python_version: str, gpu_arch: str = "gfx1102"):
    try:
        rocm_path, rocm_version = detect_rocm_installation()
        logger.info(f"Rilevato ROCm {rocm_version} in {rocm_path}")

        # Patcha il CMake di gloo prima della build
        patch_gloo_cmake(source_path, rocm_path)

        clang_path = check_clang()
        logger.info(f"Usando clang++ da: {clang_path}")

        # Prepara l'ambiente di build
        build_env = prepare_build_environment(rocm_path, rocm_version, gpu_arch)

        # Verifica la directory sorgente
        if not os.path.exists(source_path):
            raise RuntimeError(f"Directory sorgente non trovata: {source_path}")

        # Cambia alla directory sorgente e avvia la build
        os.chdir(source_path)
        logger.info("Avvio della compilazione di PyTorch...")

        result = subprocess.run(
            [python_version, 'setup.py', 'bdist_wheel'],
            env=build_env,
            check=True
        )

        # Verifica il risultato
        wheels = list(Path('dist').glob('*.whl'))
        if wheels:
            wheel_path = wheels[-1]
            logger.info(f"Build completata con successo. Wheel generato: {wheel_path}")
            return True
        else:
            logger.error("Build completata ma nessun wheel trovato")
            return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Errore durante la compilazione: {e}")
        if e.output:
            logger.error(f"Output: {e.output}")
        return False
    except Exception as e:
        logger.error(f"Errore: {e}")
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
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
    if os.path.exists(default_path):
        if os.path.islink(default_path):
            target = os.readlink(default_path)
            version = re.search(r'rocm-(\d+\.\d+\.\d+)', target)
            if version:
                return default_path, version.group(1)

        try:
            result = subprocess.run(['rocm-smi', '--showversion'],
                                    capture_output=True, text=True, check=True)
            version = re.search(r'ROCm-(\d+\.\d+\.\d+)', result.stdout)
            if version:
                return default_path, version.group(1)
        except subprocess.CalledProcessError:
            pass

        rocm_dirs = glob.glob("/opt/rocm-*")
        if rocm_dirs:
            latest_dir = sorted(rocm_dirs)[-1]
            version = re.search(r'rocm-(\d+\.\d+\.\d+)', latest_dir)
            if version:
                return default_path, version.group(1)

    raise RuntimeError("Non è stato possibile trovare un'installazione valida di ROCm")


def find_gcc_version():
    """Rileva la versione di GCC installata nel sistema."""
    try:
        gcc_output = subprocess.check_output(['gcc', '-dumpversion'],
                                             universal_newlines=True).strip()
        gcc_path = subprocess.check_output(['which', 'gcc'],
                                           universal_newlines=True).strip()
        version_parts = gcc_output.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        return major, minor, gcc_path
    except (subprocess.CalledProcessError, ValueError, IndexError):
        raise RuntimeError("Impossibile determinare la versione di GCC")


def check_dependencies():
    """Verifica la presenza delle dipendenze necessarie."""
    required_commands = ['gcc', 'g++', 'python3', 'cmake']
    missing = []

    for cmd in required_commands:
        if not shutil.which(cmd):
            missing.append(cmd)

    if missing:
        raise RuntimeError(f"Dipendenze mancanti: {', '.join(missing)}")

    gcc_major, gcc_minor, gcc_path = find_gcc_version()
    logger.info(f"Trovato GCC versione {gcc_major}.{gcc_minor} in {gcc_path}")

    cpp_base_dirs = [
        '/usr/include/c++',
        f'/usr/include/c++/{gcc_major}',
        f'/usr/include/x86_64-linux-gnu/c++/{gcc_major}'
    ]

    cpp_dirs_found = [d for d in cpp_base_dirs if os.path.exists(d)]
    if not cpp_dirs_found:
        raise RuntimeError("Non è stata trovata una directory base valida per C++")

    logger.info("Directory C++ trovate:")
    for d in cpp_dirs_found:
        logger.info(f"  - {d}")

    return {
        'gcc_version': (gcc_major, gcc_minor),
        'gcc_path': gcc_path,
        'cpp_dirs': cpp_dirs_found
    }

def check_rocm_compatibility(rocm_version):
    """Verifica la compatibilità della versione ROCm."""
    major, minor, _ = rocm_version.split('.')
    if int(major) >= 6:  # Per ROCm 6.x
        return [
            '-DROCM_VERSION_MAJOR=' + major,
            '-DROCM_VERSION_MINOR=' + minor,
            '-DROCM_VERSION=' + ''.join([major, minor, '01']),  # Format: 60301
            '-DTORCH_HIP_VERSION=' + ''.join([major, minor, '0'])  # Format: 603
        ]
    return []

def build_pytorch(source_path: str, python_version: str, gpu_arch: str = "gfx1102"):
    """Compila PyTorch per ROCm."""
    try:
        # Rileva ROCm e dipendenze
        rocm_path, rocm_version = detect_rocm_installation()
        deps_info = check_dependencies()

        # Prepara i flag del compilatore
        hipcc_flags = [
            f'"\--rocm-path={rocm_path}"',  # Aggiunto escape e quote
            f'"\--rocm-device-lib-path={rocm_path}/amdgcn/bitcode"',
            f'"\--offload-arch={gpu_arch}"',

            # Path delle include e definizioni per rocprim
            f'-I{rocm_path}/include',
            f'-I{rocm_path}/include/hip',
            f'-I{rocm_path}/include/rocm',
            '-D__HIP_ENABLE_DEVICE_MALLOC__',
            '-DROCPRIM_DISABLE_UNINITIALIZED_ARRAY',  # Disabilita l'array non inizializzato problematico
            '-DROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS=0',
            '-D__HIP_ARCH_HAS_WARP_SHUFFLE__=1',
            '-D__HIP_PLATFORM_AMD__',
            '-D__HIP_ROCclr__',
            *[f'-I{cpp_dir}' for cpp_dir in deps_info['cpp_dirs']],

            # Flag per Rocprim e CUB
            '-D__HIP_PLATFORM_AMD__=1',
            '-DROCM_VERSION_MAJOR=' + rocm_version.split('.')[0],
            '-DROCM_VERSION_MINOR=' + rocm_version.split('.')[1],
            '-DHIP_COMPILER=clang',
            '-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP',
            '-D__HIP_DISABLE_CPP_FUNCTIONS__',
            '-DROCPRIM_DEFAULT_ALLOCATOR=1',
            '--amdgpu-function-calls=false',
            '-DROCPRIM_HIP_API=1',
            '-DROCPRIM_DISABLE_HOST_NEW_DELETE=1',
            # Prevenzione operator new
            '-D__HIP_NO_NEW_DELETE__',
            '-D__host__=__attribute__((__host__))',
            '-D__device__=__attribute__((__device__))',
            '-D__forceinline__=__attribute__((always_inline)) inline',
            '-D_HAS_UNIFIED_MEMORY=1',
            # Opzioni del compilatore
            '-std=c++17',
            '-pthread',
            '-fno-gpu-rdc',
            '--offload-arch=' + gpu_arch,
            '-fPIC',
            '-Wno-deprecated-declarations',
            '-Wno-unknown-cuda-version',
            '-Wno-return-type-c-linkage',
            '-Wno-unused-result',

            '-D__HIP_PLATFORM_HCC__',
            '-DHIP_CLANG_HCC_COMPAT_MODE=1',
            '-DCAFFE2_USE_MIOPEN',
            '-DHIPBLAS_V2',
            '-DHIP_NEW_TYPE_ENUMS',
        ]

        hipcc_flags.extend(check_rocm_compatibility(rocm_version))

        # Prepara l'ambiente
        build_env = os.environ.copy()
        build_env.update({
            'MAX_JOBS': '4',
            'TORCH_CUDA_ARCH_LIST': gpu_arch,
            'CMAKE_C_COMPILER': '/usr/bin/gcc-12',
            'CMAKE_CXX_COMPILER': '/usr/bin/g++-12',
            'HIP_PLATFORM': 'amd',
            'ROCM_PATH': rocm_path,
            'HIPCC_COMPILE_FLAGS_APPEND': ' '.join(hipcc_flags),
            'HIP_COMPILER': 'clang',
            'HIPCC_VERBOSE': '1',  # Aggiunto per debug
            'PYTORCH_ROCM_ARCH': gpu_arch,
            'USE_ROCM': '1',
            'USE_CUDA': '0',
            'ROCM_HOME': rocm_path,
            'HIP_PATH': f'{rocm_path}/hip',
        })

        # Configura CMake
        cmake_args = [
            '-DCMAKE_CXX_STANDARD=17',
            f'-DHIP_COMPILER_FLAGS="--offload-arch={gpu_arch}"',
            '-DCMAKE_HIP_ARCHITECTURES=' + gpu_arch,
            f'-DROCM_PATH={rocm_path}',
            '-DHIP_COMPILER=clang',
            f'-DROCM_VERSION={rocm_version}',
            f'-DCMAKE_PREFIX_PATH={rocm_path}',
            f'-DHIP_PATH={rocm_path}/hip',
            '-DUSE_ROCM=ON',
            '-DUSE_CUDA=OFF'
        ]

        # Aggiungi i percorsi delle librerie C++ a CMAKE_ARGS
        for cpp_dir in deps_info['cpp_dirs']:
            cmake_args.append(f'-DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES={cpp_dir}')

        build_env['CMAKE_ARGS'] = ' '.join(cmake_args)

        # Cambia directory alla source path
        os.chdir(source_path)

        logger.info("Avvio della compilazione di PyTorch...")
        subprocess.run([python_version, 'setup.py', 'bdist_wheel'],
                       env=build_env, check=True)

        # Cerca il wheel generato
        wheels = list(Path('dist').glob('*.whl'))
        if wheels:
            wheel_path = wheels[-1]
            logger.info(f"Build completata con successo. Wheel generato: {wheel_path}")
            return True
        else:
            logger.error("Build completata ma nessun wheel trovato")
            return False

    except Exception as e:
        logger.error(f"Errore durante la compilazione: {e}")
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
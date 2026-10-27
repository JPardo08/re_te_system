# Recuperacion EXP.4.5 y sincronizacion conservadora

Procedimiento local del 27 de octubre de 2026. No se ha ejecutado en CESVIMA. No autoriza J13, J16, J17 ni Gold C0/C1. No convierte el checkpoint y no reinstala FlashAttention.

## Pins incorporados

Origen UIE `luyaojie/uie-base-en@966f8b1fc4c74e94ab552081605913ad5133cc41`, fichero `pytorch_model.bin`:

```text
size    990285657
sha256  04baf49c28f4be89424f3a61dd30d674c4ab165ff16c1b007525829a8c9441f7
```

Ese digest identifica los bytes medidos en CESVIMA. `DESERIALIZATION_AUTHORIZED` permanece false. PyTorch 2.5.1 no es un runtime valido para abrir ese pickle. La puerta de Transformers 4.57.6 no se desactiva.

Wheel preservado de FlashAttention:

```text
flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl
sha256  8238e887fec100cbead2fb662a041bf8486499df00d8d026f234be3a7af2be82
```

`BUILD_PASS` nombra esa compilacion, terminada a las 19:36 UTC del 26/10. La extension que hay dentro de `gollie-l40s` es anterior: SHA-256 `1fed48745de90c41071d324ff0bbdf17c3c054c679279e5bc30c640b74007a03`, instalada a las 14:50 UTC del mismo dia, con requisito hasta `GLIBC_2.32`. La inspeccion estatica del wheel corregido da GLIBC maximo 2.14. Las dos piezas declaran FlashAttention `2.8.3.post2` y no comparten identidad binaria. El wheel corregido no se ha ejecutado en L40S. Nada de eso es `ABI_PASS`: `abi_pass` queda false, `abi_failed` queda false y `abi_verdict` queda `not_decided`. El maximo GLIBC estatico no decide la ABI.

## Lo que el codigo hace ahora

UIE. J12 hashea el `.bin` si el puntero de J11 esta presente y escribe el informe en `uie/preflight/jobs/<SLURM_JOB_ID>/`. No reescribe `uie/preflight/preflight_report.json`. No llama a `from_pretrained` mientras la conversion sigue deshabilitada. El fingerprint, cuando haya un derivado, incluye los cuatro hashes del 10-shot y el SHA-256 de `model.safetensors`. J13 rechaza un preflight fallido, incompleto o de otro JOBID, y despues rechaza el entrenamiento porque `UIE_J13_TRAINING_AUTHORIZED` es false.

La conversion excepcional esta descrita en `scripts/cesvima/uie_safetensors_conversion.py` y sale antes de cualquier loader. El diseno, para una autorizacion futura y distinta, es un entorno independiente, PyTorch `>=2.6`, `weights_only=True`, sin escribir dentro del snapshot de Hugging Face, y con comprobacion de nombres, formas, dtypes, valores, almacenamientos compartidos y equivalencia de la carga. `scripts/cesvima/uie_recovery_phases.py` ordena las fases sin activarlas: pin identificado, deserializacion explicita, conversion terminada, equivalencia del derivado, permiso de J12 para cargar solo ese derivado, batch con `PREFLIGHT_PASS`, y solo despues una autorizacion humana de J13. Un paso anterior no enciende el siguiente. Los flags vivos siguen en false. Poner `CONVERSION_ENABLED` y el permiso de carga de J12 no acepta el derivado: la equivalencia se recalcula desde los registros de tensores y el SHA-256 medido de `model.safetensors`. Un booleano no sustituye esos registros. PyTorch `2.5.1` no abre el origen y el control de Transformers tiene que seguir activo.

`run_uie_finetune.py` del clon upstream, linea 188, sigue pasando `mirror='tuna'` a `AutoModelForSeq2SeqLM.from_pretrained`. El mismo argumento aparece en `run_uie_pretrain.py`. No se ha editado el clon. En esta maquina no hay un Transformers importable, asi que la compatibilidad con 4.57.6 sigue `unverified_against_transformers_4.57.6`. J12 y J13 no reenvian `mirror`. J13 no llega al trainer.

GoLLIE, transiciones. `BUILD_PASS` es la compilacion del wheel y no acredita `gollie-l40s`. La identidad binaria compara version, SHA-256 del wheel y SHA-256 de la extension contenida en el. El hash `1fed4874…` queda fuera de `MATCH` aunque alguien afirme que el wheel lo contiene. La extension acreditada es exactamente `lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so`. Una copia solo dentro de `flash_attn/`, u otra copia al lado de esa, no se elige. `ABI_PASS` exige, ademas de `MATCH`, tres observaciones en el nodo L40S: `ldd` sin bibliotecas ausentes, simbolos `GLIBC_*` cubiertos por la libc en ejecucion, e importacion del paquete sin lanzar el kernel. El GLIBC 2.14 estatico del wheel no es una de esas observaciones. El kernel minimo solo corre despues de esa ABI. `PREFLIGHT_PASS` exige las seis condiciones y el JOBID del informe. Hoy `current_artifacts_allow_preflight` es false. Un `PREFLIGHT_PASS` futuro no pone `GOLLIE_J16_DOWNLOAD_AUTHORIZED`. `execute_isolated_recovery()` sigue sin ejecutarse. La preparacion manual de un venv nuevo es `bootstrap_gollie_isolated_venv.sh` y no se ha lanzado; no modifica `gollie-l40s`.

`bootstrap_gollie_venv.sh` no contiene `pip` en codigo ejecutable. Si el venv no existe, sale 1. Si existe, evalua versiones y la identidad de la extension. Exit 0 significa `MATCH` de la extension con el wheel pineado. No significa `ABI_PASS` ni `PREFLIGHT_PASS`. Exit 2 cubre `BINARY_MISMATCH` y comparacion ausente, sin reinstalar y sin compilar. J15 no lanza el kernel si la identidad o la ABI no estan acreditadas. Escribe el informe en un directorio de JOBID y solo entonces, con `PREFLIGHT_PASS`, el puntero de ese JOBID. J16 ignora `gollie/preflight_report.json`, rechaza otro JOBID y, aun cuando `j16_rejection_reason` no encuentra un defecto tecnico, no descarga porque `GOLLIE_J16_DOWNLOAD_AUTHORIZED` es false. J17 sale si `GOLLIE_J17_SMOKE_AUTHORIZED` no vale `1`. `env.sh` no define esa variable.

`env.sh` y `j00_verify_infer_env.py` no forman parte de este cambio. En el historial local son identicos entre `a65fb07` y `67205d4`. Las modificaciones de esos dos ficheros en el checkout remoto son trabajo que hay que preservar y revisar para una posible integracion; no se descartan y no se sobrescriben.

## Sincronizacion, sin ejecutarla

Estado de partida de aquella sincronizacion: CESVIMA en `a65fb07`, con `env.sh` y `j00_verify_infer_env.py` modificados y con scripts y logs historicos sin seguimiento. Ese checkout se sincronizo despues a mano a `17c23a69f5633432334e50e2988bead9e5f8698b`, conservando las modificaciones locales de `env.sh` y J00. La correccion de descubrimiento y ABI de J15 esta en `d1650c3`. El usuario confirma que ese codigo ya esta sincronizado y que `env.sh` conserva sus adaptaciones locales. Este procedimiento no edita `env.sh`.

1. En el nodo de login, inventario: `git status`, `git diff` de esos dos ficheros y lista de no seguidos. Sin `clean`, sin `reset`, sin `stash`, sin `checkout` de esos paths.
2. Copiar fuera del repositorio, a un directorio de preservacion en SCRATCH, los dos ficheros, el diff y los no seguidos. Verificar SHA-256 de las copias antes de seguir.
3. Solo cuando el investigador haya revisado ese diff, y solo porque las copias ya estan verificadas, dejar el arbol de trabajo en un estado en el que Git pueda avanzar en fast-forward. Si no puede, parar.
4. El avance trae `67205d4` y, cuando existan, los commits de esta recuperacion. No toca venvs, caches, el arbol aislado de Qwen3 ni `runs/`.
5. Reaplicar el diff preservado de `env.sh` y `j00_verify_infer_env.py` solo si sigue haciendo falta. Si no aplica limpio, parar. No elegir automaticamente la copia remota ni la local. Esa integracion, si se acepta, es un commit propio y revisado, no un efecto del fast-forward.

## Despliegue manual posterior, sin ejecutarlo

No lanzar este despliegue en la misma sesion que la sincronizacion si el diff de `env.sh` o de J00 no esta ya preservado.

1. Confirmar que el checkout remoto contiene el commit de recuperacion y que `git status` no muestra cambios sin copiar.
2. No borrar `uie-train` ni `gollie-l40s`. No ejecutar `pip install`. No pasar `GOLLIE_BOOTSTRAP_ALLOW_COMPILE`.
3. Lectura de UIE, sin conversion: el informe nuevo de un J12, si algun dia se autoriza el job, tiene que salir `FAIL` con `deserialization_authorized=false` y el pin `04baf49c…`. El `preflight_report.json` antiguo permanece.
4. Lectura de GoLLIE, sin reinstalacion: el bootstrap tiene que salir 2 mientras la extension siga en `BINARY_MISMATCH`. J15 puede escribir `BLOCKED_BINARY_IDENTITY` con `abi_failed=false`. Eso no es `PREFLIGHT_PASS` y no escribe el puntero que J16 aceptaria.
5. Una recuperacion aislada del wheel, si se autoriza en otra decision, usa un venv nuevo y solo el wheel cuyo SHA-256 sea `8238e887…`. No recompila y no pisa `gollie-l40s` en el mismo paso.
6. No enviar J13, J16, J17, `--execute-c0-c1` ni la conversion.

## Transiciones que el codigo admite y que hoy estan cerradas

GoLLIE, en orden. Cada fila solo puede avanzar si la anterior de esa misma ejecucion es cierta. La evidencia del 26/10 deja la instalacion actual en la segunda fila.

| Estado | Que lo abre | Estado de los artefactos actuales |
|---|---|---|
| BUILD_PASS | Wheel `8238e887…` compilado y conservado | Cierto para el wheel. No para `gollie-l40s` |
| Identidad MATCH | Version `2.8.3.post2`, SHA del wheel y SHA de su extension, iguales a la extension instalada | `BINARY_MISMATCH`. El instalado es `1fed4874…`, GLIBC 2.32, 14:50 UTC |
| ABI_PASS | MATCH, dependencias dinamicas resueltas y GLIBC presente, en el nodo GPU | `not_decided`. El GLIBC 2.14 estatico no basta |
| Kernel L40S | ABI_PASS y kernel minimo finito en L40S | No ejecutado para el wheel corregido |
| PREFLIGHT_PASS | Las cuatro anteriores, nodo e informe de ese JOBID | Inalcanzable con estos artefactos |
| J16 | Decision explicita distinta de PREFLIGHT_PASS | `GOLLIE_J16_DOWNLOAD_AUTHORIZED=false` |

UIE, en orden. Identificar el pin no abre la deserializacion. Un derivado valido puede pasar la identidad y seguir sin permiso de carga. J12 PASS no pone el flag de J13.

| Fase | Que la abre | Flag vivo |
|---|---|---|
| Origen identificado | `pytorch_model.bin`, 990285657 bytes, SHA `04baf49c…` | Pin registrado. No es un `torch.load` |
| Deserializacion excepcional | Autorizacion explicita, entorno aislado, torch `>=2.6`, control de Transformers activo | `DESERIALIZATION_AUTHORIZED=false` |
| Conversion terminada | La fase anterior y un safetensors fuera del snapshot | `CONVERSION_ENABLED=false` |
| Equivalencia | Nombres, formas, dtypes, valores, almacenamiento compartido y carga del derivado | Sin artefactos reales |
| J12 carga el derivado | Equivalencia y `UIE_J12_DERIVED_LOAD_AUTHORIZED` | false. J12 sale antes de `from_pretrained` |
| J12 PREFLIGHT_PASS | Batch real sobre ese derivado, informe de ese JOBID | No ejecutado |
| J13 | Autorizacion humana posterior | `UIE_J13_TRAINING_AUTHORIZED=false` |

## Preparacion aislada de GoLLIE, sin ejecutarla

R1 es `scripts/cesvima/bootstrap_gollie_isolated_venv.sh`. El modo por defecto es `inventory`: comprueba el SHA-256 del wheel y si el directorio de wheels contiene una sola pieza de torch `2.5.1+cu121`, transformers `4.57.6`, black `24.10.0`, triton `3.1.0`, einops `0.8.2` y huggingface_hub `0.36.2`. No crea el venv y no llama a pip. `install` sale inmediatamente: R1 Install no esta autorizado aunque `GOLLIE_ISOLATED_INSTALL=1`. `resolve` es solo el dry-run, con `--ignore-installed`, `--no-index` y `--only-binary=:all:`, y no escribe la cache. No hay `--index-url`, ni compilacion, ni copia de `gollie-l40s`. El destino previsto sigue siendo `${SCRATCH_ROOT}/venvs/gollie-isolated`.

En CESVIMA hay 103 wheels en `pip-cache/http-v2`, de 86 distribuciones. El inventario anterior nombro versiones. `missing` es esa lista estatica, con `selection_executed` false: las diez dependencias NVIDIA `==` de CUDA 12.1 siguen `MISSING` y `selected_candidate` false. No es la prueba de una seleccion. `nvidia-cudnn-cu12==9.1.0.70` y las seis piezas del stack quedan `VERSION_OBSERVED`. `sympy==1.13.1` tambien estaba en esa observacion de version. `sympy==1.14.0` sigue `INCOMPATIBLE`. El METADATA del wheel oficial `torch-2.5.1+cu121` declara esas diez dependencias y `setuptools` sin version para Python 3.12. Esas diez no son los pines 12.4 de `torch 2.5.1` en PyPI. `nvidia-nvjitlink-cu12` aparece sin version, via cusolver y cusparse, y no se elige una version.

`select-cache` en CESVIMA termino con EXIT=0. El manifiesto tiene SHA-256 `e73bc2dc212fb31324eb0b07c4f1c9e0dfeb0f078d783dd4fafcb0ba23883fe0`. Conteos: 7 seleccionados, 0 ambiguos, 21 rechazados y 77 sin pin exacto. `materialized` e `installable_wheelhouse` salieron false. `sympy==1.13.1` quedo como candidato seleccionado, con SHA-256 medido `db36cdc64bf61b9b24578b6f7bab1ecdd2452cf008f34faa33776680c26d66f8`, sin procedencia acreditada. El mismo paso rechazo `nvidia-cudnn-cu12==9.1.0.70`, SHA-256 medido `165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f`, tag `py3-none-manylinux2014_x86_64`. Ese tag es compatible con CPython 3.12, Linux x86_64 y glibc 2.28; el rechazo vino del predicado anterior. La compatibilidad no acredita procedencia. `sympy==1.14.0`, torch 2.14 y triton 3.8 siguen rechazados. Los nombres de los otros seis seleccionados no constan aqui.

El selector usa `packaging.tags` con plataformas generadas para Linux x86_64 y glibc 2.28. No consulta la plataforma del Mac. Distingue `cp312-cp312`, `cp39-abi3`, `py3-none-any` y `py3-none-manylinux2014_x86_64`. No admite macOS, Windows, aarch64 ni un manylinux que exija glibc superior a 2.28.

`select-cache` solo escribe el manifiesto. La pasada posterior, ya con el tag `manylinux2014` aceptado, dejo el manifiesto `$SCRATCH_ROOT/recovery/gollie-wheel-manifest-exp472-20261027.json`, SHA-256 `7308b6ae7c4aa7df5a324b5e057c9c8f655ee958568cc8716e208ca3ebab8357`. Conteos de ese fichero: 8 seleccionados, 20 rechazados, 0 ambiguos, `materialized` false e `installable_wheelhouse` false. `nvidia-cudnn-cu12==9.1.0.70` quedo seleccionado. Las diez dependencias NVIDIA de CUDA 12.1 siguen ausentes. Los hashes por fichero de ese manifiesto no se copiaron a este repositorio.

`materialize` lee ese manifiesto y comprueba su SHA-256. Despues vuelve a leer cada origen dentro de la cache, recomputa el SHA-256 y compara el METADATA y los tags releidos con el registro. El nombre y la version declarados dentro del ZIP no bastan. La procedencia queda acreditada solo si ese SHA-256 coincide con un unico digest publicado y el nombre PEP 427 de ese fichero coincide con el METADATA y los tags. El digest publicado de cuDNN `9.1.0.70` es `165764f44ef8c61fcdfdfdbe769d687e06374059fbb388b6c89ecb0e28793a6f`, el de `torch-2.5.1+cu121-cp312-cp312-linux_x86_64.whl` es `222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76`, y hay digests de PyPI para transformers `4.57.6`, black `24.10.0`, triton `3.1.0`, einops `0.8.2`, huggingface_hub `0.36.2` y sympy `1.13.1`. Un candidato sin digest publicado no se copia y queda en `blocked` con el motivo. La copia va a un directorio nuevo dentro de `SCRATCH_ROOT`, fuera de la cache y de los venv. No sobrescribe, no acepta dos filenames iguales, y verifica el SHA-256 antes y despues. Deja `materialization.json` y `SHA256SUMS`. La cache no se modifica.

Esa copia no cierra el entorno. `setuptools` sigue sin version para Python 3.12 y `nvidia-nvjitlink-cu12` sigue sin version, exigido por cusolver `11.4.5.107` y cusparse `12.1.0.106`. No se leyeron sus dependencias porque no hay una version acreditada. Las diez ruedas CUDA 12 ausentes conservan los URL y SHA-256 del plan, y `download` no se ejecuta. `resolve` mantiene `--dry-run --ignore-installed --no-index --only-binary=:all:` y no se declara un cierre hasta que ese dry-run termine bien. `installable_wheelhouse` permanece false. R1 Install, R2, J16, J17 y Gold siguen cerrados. El wheel compilado de FlashAttention no se recompila. `materialize` exige `GOLLIE_MATERIALIZE_AUTHORIZED=1`; sin esa variable sale antes de copiar.

R2 es `jobs/cesvima/j15_gollie_isolated_preflight.sbatch`. Sale al empezar salvo `GOLLIE_R2_PREFLIGHT_AUTHORIZED=1`, valor que `env.sh` no define. Si algun dia se abre, carga `env.sh` y despues asigna `GOLLIE_VENV` al venv aislado. Exige el registro `PREP_PASS`, el fichero local `modeling_flash_llama.py` de la revision `d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9` y `HF_HUB_OFFLINE=1`. No descarga pesos. El kernel sigue siendo el de J15, solo en L40S y despues de identidad y ABI. El informe sigue siendo por JOBID. `submit_train_first.sh` no envia este sbatch. J16, J17 y Gold siguen cerrados.

## Commits propuestos, sin crearlos

1. `gollie: pin the FlashAttention wheel and keep the pre-correction binary unaccredited`
2. `uie: pin the origin checkpoint and keep every later phase disabled`
3. `docs: record the EXP.4.5.1 transition gates and the conservative sync`

El primero cubre `flash_attn_pin.py`, `gollie_environment.py`, el bootstrap, J15, J16, J17, `tests/test_exp45_gollie_recovery.py` y `tests/test_exp451_gollie_transitions.py`. El segundo cubre el pin de origen, la conversion deshabilitada, `uie_recovery_phases.py`, el fingerprint, J12, J13, `tests/test_exp45_uie_recovery.py` y `tests/test_exp451_uie_transitions.py`. El fichero mixto `tests/test_exp45_recovery.py` ya no existe: sus negativos quedaron partidos en esos dos ficheros. El tercero cubre este fichero y `docs/CESVIMA_TRAIN_FIRST.md`. No incluyen manifiestos cientificos ni el clon upstream. Esos commits ya estan en el historial local que termina en `d1650c3`. La preparacion aislada posterior esta en `1f70e97`, la clasificacion de la cache en `2f0aeef` y la correccion del tag en `096be88`. El manifiesto `7308b6ae` es la medicion de EXP.4.7.2 en CESVIMA.

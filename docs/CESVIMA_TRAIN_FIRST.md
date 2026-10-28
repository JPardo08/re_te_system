# CESVIMA train-first

Capa versionable de jobs para Paper 3 en Magerit. Esta capa no es un experimento C0/C1. No descarga modelos ni entrena nada por el hecho de existir en el repositorio: solo describe jobs que se enviaran cuando alguien ejecute el submitter en CESVIMA.

El camino soportado de envio es `scripts/cesvima/submit_train_first.sh`. Un `sbatch` manual del fichero suelto puede dejar `slurm-*.out` en el directorio de trabajo. El submitter redirige stdout y stderr a SCRATCH.

## Arquitectura del DAG

Todas las rutas remotas salen de `scripts/cesvima/env.sh`. Los `#SBATCH` no expanden variables de shell, asi que particion, GRES, tiempo, CPUs y memoria van literales en cada job. Los paths no.

```text
J00 verify infer venv
  J01 mREBEL cache -> J02 mREBEL smoke3 -> J03 mREBEL full
  J04 REBEL resolve SHA -> J05 REBEL download
  J11 UIE base cache
  J08 UIE fetch upstreams
    J00 + J08 -> J09 rebuild -> J10 freeze example
    J09 + J11 -> J12 preflight -> J13 train seed1/10shot
    J13 + J10 -> J14 positive smoke
J15 GoLLIE preflight L40S -> J16 download 7B -> J17 native smoke
J18 GenIE ckpt
J00 -> J19 GenIE tokenizer
J20 GenIE tries
```

J06 existe, pero no entra en este DAG: `BLOCKED_FIXTURE`. J07 es el diagnostico cruzado Hohfeld y solo se envia con `--rebel-cross-lingual`.

Dos aristas siguen mas apretadas que el diagrama de la auditoria, sin cambiar el entrenamiento UIE:

- J09 espera a J00 ademas de a J08, porque la reconstruccion usa el interprete ya verificado.
- J19 espera a J00, porque la descarga del tokenizer usa `huggingface_hub` del venv de inferencia. J18 y J20 son `curl` mas verificacion de tamano y MD5, y no necesitan ese venv.

GoLLIE no depende de J00. J15 crea o reutiliza `gollie-l40s`. `--all` envia J00 una sola vez y no envia J06 ni J07.

## Politica de lengua

Hay tres estratos. No son equivalentes y no comparten ranking.

- `SPANISH_HOHFELD_MAIN`. Sistemas con soporte justificable de espanol, y el futuro backbone controlado C0-C3. En esta fase el unico sistema de este estrato es mREBEL.
- `NATIVE_REPRODUCTION`. Cada sistema se reproduce en su lengua y tarea documentadas. REBEL es ingles. UIE es ingles sobre CoNLL04. GoLLIE es ingles sobre el fixture RE oficial. GenIE sera ingles cuando exista loader. Pythia sera ingles/SPACE-KBP cuando se resuelva la licencia.
- `CROSS_LINGUAL_DIAGNOSTIC`. Un sistema English-only puede ejecutarse sobre Hohfeld espanol para estudiar transferencia. Ese resultado no entra en el ranking principal, no se interpreta como comparacion equivalente, y queda marcado `out_of_documented_language_scope` con `principal_ranking=excluded`.

## Que entrena y que no

Entrena solo J13: UIE `luyaojie/uie-base-en` sobre CoNLL04, seed1, 10-shot. Eso es `NATIVE_REPRODUCTION`, no una adaptacion al espanol ni a Hohfeld Gold. No hay entrenamiento UIE sobre Hohfeld. Un eventual UIE sobre Hohfeld espanol seria `CROSS_LINGUAL_DIAGNOSTIC` y no forma parte de esta tanda.

No se lanzan `run_uie_finetune_shot.bash`, `scripts_exp/run_exp_shot.bash` ni `scripts/function_code.bash`. No se ejecutan `sel2record.py` ni `eval_extraction.py`.

No entrenan: mREBEL, REBEL, GoLLIE, GenIE ni Pythia. C0 y C1 no estan en este DAG. Priority B permanece en `STANDBY`.

GoLLIE es inferencia local-only, `quantization=none`, sin fine-tuning, sobre el fixture ingles oficial. GenIE solo adquiere tres artefactos: estado experimental `TARGET / BLOCKED_LOADER`. No hay job de inferencia GenIE. Pythia sigue siendo un sistema objetivo: `TARGET / BLOCKED_LICENSE`. No se descarga ni se ejecuta el checkpoint fine-tuned hasta resolver la licencia. La licencia pendiente no lo saca de Priority A.

Los derivados CoNLL04 existen en el workspace local:

```text
_legacy/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04/
_legacy/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04_shot/seed1/{1shot,5shot,10shot}/
_legacy/sincere/data/conll04.json
```

Esos derivados estan ignorados por git. No se copian a CESVIMA. J09 los reconstruye en SCRATCH desde los clones pineados y valida los hashes contra el freeze versionado. Solo `seed1/10shot` entra en J13. `1shot` y `5shot` son derivados historicos y no se ejecutan.

## Tabla J00-J20

| Job | Que hace | Particion | GRES | Tiempo | CPUs | Memoria |
| --- | --- | --- | --- | --- | --- | --- |
| J00 | Verifica el venv de inferencia. No crea otro. | standard | | 00:20:00 | 2 | 8G |
| J01 | Cache o descarga mREBEL pineado. | standard | | 02:00:00 | 2 | 8G |
| J02 | Smoke3 mREBEL, local-only, V100. | standard-gpu | gpu:v100:1 | 04:00:00 | 4 | 32G |
| J03 | Full mREBEL, 36 documentos, local-only, V100. | standard-gpu | gpu:v100:1 | 12:00:00 | 4 | 32G |
| J04 | Resuelve el SHA actual de REBEL y lo escribe. | standard | | 00:30:00 | 1 | 4G |
| J05 | Rechaza una revision que no sea SHA y descarga ese SHA. | standard | | 02:00:00 | 2 | 8G |
| J06 | Smoke nativo ingles. Hoy `BLOCKED_FIXTURE`: no infiere y no lee Hohfeld. | standard | | 00:10:00 | 1 | 4G |
| J07 | Diagnostico cruzado Hohfeld espanol. Fuera de `--all`. | standard-gpu | gpu:v100:1 | 12:00:00 | 4 | 32G |
| J08 | Clona UIE y sincere en los SHA pineados. | standard | | 01:00:00 | 2 | 8G |
| J09 | Reconstruye CoNLL04 en SCRATCH y compara hashes. | standard | | 04:00:00 | 4 | 16G |
| J10 | Elige un ejemplo de test antes de cualquier prediccion. | standard | | 00:30:00 | 1 | 4G |
| J11 | Cache o descarga el checkpoint base de UIE. | standard | | 02:00:00 | 2 | 8G |
| J12 | Preflight de un batch real en el venv `uie-train`. | standard-gpu | gpu:v100:1 | 04:00:00 | 4 | 32G |
| J13 | Entrena seed1/10shot si el fingerprint de J12 coincide. | standard-gpu | gpu:v100:1 | 48:00:00 | 8 | 32G |
| J14 | Smoke positivo del best checkpoint por el adapter limpio. | standard-gpu | gpu:v100:1 | 02:00:00 | 4 | 32G |
| J15 | Preflight L40S. No descarga los pesos 7B. | standard-gpu | gpu:l40s:1 | 06:00:00 | 8 | 64G |
| J16 | Descarga GoLLIE-7B despues de J15. | standard | | 06:00:00 | 2 | 8G |
| J17 | Smoke nativo local-only con los fixtures oficiales. | standard-gpu | gpu:l40s:1 | 04:00:00 | 8 | 64G |
| J18 | Descarga y verifica `genie_r.ckpt`. | standard | | 04:00:00 | 2 | 8G |
| J19 | Descarga el tokenizer GenIE pineado. | standard | | 01:00:00 | 2 | 8G |
| J20 | Descarga y verifica `tries.zip`. | standard | | 02:00:00 | 2 | 8G |

Los tiempos son presupuesto operativo, no hiperparametros. La documentacion publica de Magerit permite hasta 72 h en `standard-gpu` a recursos completos y un maximo absoluto de 160 h. J13 pide 48 h.

V100: mREBEL, REBEL J07 y UIE. L40S: GoLLIE. CPU: adquisicion, hashes, reconstruccion, validacion y el J06 bloqueado. No hay A100. No hay `--gres=gpu:N` sin tipo. Ningun job cambia de particion si queda `PD`.

## Recursos y entornos

Venv canonico de inferencia, verificado, solo comprobacion en J00:

```text
/home/y064/SCRATCH/y064644/paper3/venvs/re_te_system_gpu_312
```

J00 exige Python 3.12.3, torch 2.5.1+cu121, torch CUDA 12.1, transformers 4.57.6, sentencepiece 0.2.2, y que `re_te_system` se importe desde `PROJECT/re_te_system/src` como instalacion editable. Si falta algo, el job sale con estado distinto de cero y un diagnostico. No crea un venv general y no crea `paper3-infer`.

`/home/y064/SCRATCH/y064644/paper3/venvs/re_te_system_312` se conserva. No se usa para GPU por la incompatibilidad CUDA ya observada. J00 falla si el interprete sale de ese venv. No se borra.

`uie-train` sigue siendo el venv de entrenamiento UIE, porque la compatibilidad del trainer upstream es un problema distinto del runtime de inferencia. `gollie-l40s` sigue siendo el venv de GoLLIE. Los dos viven bajo `SCRATCH_ROOT/venvs/`. El bootstrap de UIE crea `uie-train` solo si `bin/python` no existe. Reutilizar ese venv no autoriza a abrir `pytorch_model.bin` con `torch.load`. El bootstrap de GoLLIE ya no crea el venv y ya no compila: valida el entorno existente y sale en fallo si la extension instalada no acredita el wheel pineado. No hace `pip install` dentro de `re_te_system_gpu_312`. GoLLIE usa `PYTHONPATH=$SYSTEM_ROOT/src` en J17 para no instalar el paquete en editable dentro del repo.

J15 no compila `flash-attn`. El wheel preservado es `flash-attn==2.8.3.post2`, SHA-256 `8238e887fec100cbead2fb662a041bf8486499df00d8d026f234be3a7af2be82`. La comparacion observada de `flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so` es `BINARY_MISMATCH`: el paquete instalado no se trata como ese wheel, y ese resultado no es un veredicto de ABI. Si `nvcc` falta, no se inventa un modulo Lmod ni una compilacion de reemplazo.

## Gates

- J00 FAIL si el venv existente no coincide con el pin.
- J02/J03 FAIL si `resolved_revision` no es el SHA pedido, si `relation_filter` no es `none`, si `inputs_succeeded` no es 3 o 36, si `inputs_failed` no es 0, o si falta uno de los cuatro artefactos. F1 no es criterio de PASS.
- J04 escribe repo, SHA, timestamp UTC y fuente `huggingface_hub.HfApi.model_info`. No inventa el SHA.
- J05 rechaza cualquier revision que no sea 40 hexadecimales en minuscula.
- J06 sale 2 con `BLOCKED_FIXTURE`. No lee `documents.jsonl` ni los tres articulos de smoke. No hay en el repositorio un documento ingles oficial de REBEL versionado: los tests de parser cubren cadenas de salida, y el pipeline mock usa una frase espanola. No se inventa gold.
- J07 usa el canonico Hohfeld de 36 documentos, en espanol y sin traducir. El manifiesto debe quedar en `CROSS_LINGUAL_DIAGNOSTIC`, `language_status=out_of_documented_language_scope` y `principal_ranking=excluded`. No entra en `--all`.
- J09 FAIL si un hash no coincide con el manifiesto versionado. No copia los arboles ignorados de `_legacy`, aunque esos derivados existen localmente. La reconstruccion original se hizo con Python 3.8; aqui el interprete es el 3.12 verificado. Un mismatch de hash es un fallo cerrado.
- J10 elige la primera linea de test cuyo texto no esta en train. Guarda identidad y SHA. No lee predicciones.
- J12 no abre `pytorch_model.bin`. El pin de origen es SHA-256 `04baf49c28f4be89424f3a61dd30d674c4ab165ff16c1b007525829a8c9441f7` y 990285657 bytes; coincidir con ese pin no autoriza la deserializacion. La conversion a safetensors permanece deshabilitada. J12 escribe el informe en `uie/preflight/jobs/<SLURM_JOB_ID>/` y no reescribe `uie/preflight/preflight_report.json`. No parchea el trainer ni la puerta de seguridad de Transformers.
- J13 rechaza un preflight fallido, incompleto o sin JOBID, y tambien un informe de otro JOBID. Aunque el informe estuviera completo, `UIE_J13_TRAINING_AUTHORIZED` permanece false y el entrenador no arranca. El recalculo del fingerprint, cuando esa autorizacion exista, sigue abortando si difiere del de J12. Solo acepta un directorio `seed1/10shot`. Si el directorio de salida ya tiene ficheros, aborta; un requeue no reanuda un entrenamiento a medias.
- J14 PASS tecnico si `parsed_structures` no esta vacio y la procedencia del checkpoint no usa el SHA base como revision de los pesos. La recuperacion de la relacion gold se informa aparte y no decide ella sola el exit code. Si la estructura esta vacia, o si `resolved_revision` es `966f8b1fc4c74e94ab552081605913ad5133cc41`, el exit es distinto de cero.
- J15 no alcanza `PREFLIGHT_PASS` mientras la identidad binaria no sea `MATCH`. Un `BINARY_MISMATCH` deja `abi_pass` en false y `abi_verdict` en `not_decided`. El informe va a `gollie/preflight/jobs/<SLURM_JOB_ID>/` e incluye el JOBID y el SHA-256 del wheel. El JSON antiguo `gollie/preflight_report.json` no se reescribe y no autoriza nada. Siguen vigentes la L40S, `torch==2.5.1+cu121`, el matmul bfloat16, Black `>=24.8,<25`, el kernel minimo y el import de `modeling_flash_llama.py`, pero solo despues de acreditar el binario. Esos 40 GiB libres siguen siendo un umbral operativo: el unico dato de memoria sustanciado en el repo es 13477.10 MB de notebook, no un tamano hasheado del checkpoint 7B.
- J16 no descarga si el puntero de JOBID falta, si el informe es de otro JOBID, si el estado no es `PREFLIGHT_PASS`, si la identidad binaria no es `MATCH`, si faltan `BUILD_PASS` o `ABI_PASS`, o si `weights_downloaded` no es false. El booleano `GOLLIE_J16_DOWNLOAD_AUTHORIZED` del modulo permanece false. La descarga solo arranca si el entorno del job trae esa variable igual a `1`, el interprete es `gollie-isolated` y no `gollie-l40s`, hay al menos 40 GiB libres en SCRATCH, `HF_HUB_CACHE` es escribible dentro de SCRATCH y la metadata del repositorio responde sin 401 ni 403. Si el directorio local existe pero le faltan archivos, J16 no usa `cache_or_download`: ese helper devolveria el directorio parcial sin completarlo. Pide el snapshot con `local_files_only` desactivado y solo entonces exige `config.json`, `modeling_flash_llama.py` de 46523 bytes, el indice si existe y todos los pesos que referencia. Un resultado que sigue incompleto no escribe `DOWNLOAD_PASS`. El informe queda en `gollie/download/jobs/<SLURM_JOB_ID>/`. No usa `--all`.
- J17 sigue bloqueado salvo `GOLLIE_J17_SMOKE_AUTHORIZED=1` en el entorno del job, valor que `env.sh` no define, un informe `DOWNLOAD_PASS` de J16 y un informe `CONVERSION_PASS` para `HiTZ/GoLLIE-7B@d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9`. La conversion es un job CPU aparte, con un interprete que no es `gollie-isolated` ni `gollie-l40s` y con PyTorch >=2.6. Lee solo los shards de `pytorch_model.bin.index.json` mediante `torch.load(..., map_location="cpu", weights_only=True)` y escribe Safetensors en `P3_MODELS_ROOT`, sin tocar la cache ni los `.bin`. Esos dos shards suman 13477726551 bytes. El smoke carga el derivado con `use_safetensors=True` y no vuelve a los `.bin`. El sbatch de inferencia apunta `GOLLIE_VENV` a `gollie-isolated` despues de cargar `env.sh`. El criterio tecnico sigue siendo cuatro artefactos, `inputs_succeeded==1` y esa revision. Una generacion vacia no es, por si sola, el criterio de PASS. El stdout del CLI solo vale si su ultima linea no vacia es un unico objeto JSON con `run_dir`; las lineas anteriores se conservan en el log y una salida ambigua o malformada falla. `jobs/cesvima/j17_gollie_trace.sbatch` no entra en `submit_train_first.sh`. Solo si el entorno trae la traza autorizada, una excepcion de extraccion deja `extraction_traceback.txt` y el traceback en stderr, sin anadir campos a `failures.jsonl` ni relajar ese criterio.
- J18 y J20 reutilizan el fichero si tamano y MD5 coinciden. Si el tamano no coincide, borran el parcial en SCRATCH y vuelven a descargar. No deserializan el checkpoint ni el zip.
- Una licencia Hugging Face que responda 401 o 403 deja el job en fallo explicito. `HF_TOKEN`, si hace falta, sale del entorno. No se commitea.

## Paths

```text
PROJECT_ROOT=/home/y064/PROJECT/y064644/paper3
SCRATCH_ROOT=/home/y064/SCRATCH/y064644/paper3
SYSTEM_ROOT=$PROJECT_ROOT/re_te_system
BENCHMARK_ROOT=$PROJECT_ROOT/re_te_benchmark
EVALUATION_ROOT=$PROJECT_ROOT/re_te_evaluation
HF_HOME=$SCRATCH_ROOT/hf-cache
HF_HUB_CACHE=$SCRATCH_ROOT/hf-cache/hub
PIP_CACHE_DIR=$SCRATCH_ROOT/pip-cache
TMPDIR=$SCRATCH_ROOT/tmp
PYTHONUNBUFFERED=1
```

Pesos, caches, runs y checkpoints van a SCRATCH. El prefijo de runs es `$SCRATCH_ROOT/runs/p3`. Los logs de SLURM van a `$SCRATCH_ROOT/runs/p3/slurm`. Los upstreams clonados van a `$SCRATCH_ROOT/upstreams`. GenIE va a `$SCRATCH_ROOT/models/genie`.

`cesvima_guard` rechaza un entorno cuya particion o GRES mencione A100, un `SCRATCH_ROOT` dentro de `PROJECT_ROOT`, o un `HF_HOME` dentro de los tres repos.

## Inputs y outputs

| Job | Input principal | Output en SCRATCH |
| --- | --- | --- |
| J00 | venv existente | diagnostico en el log |
| J01 | `Babelscape/mrebel-large@50e0587ac7ac87d28b9abd069d72333528a5aa09` | `runs/p3/mrebel/snapshot.json` |
| J02 | `articulo_61`, `articulo_8`, `articulo_12` | `runs/p3/mrebel/smoke3/` |
| J03 | `documents.jsonl` canonico, 36 lineas | `runs/p3/mrebel/full/` |
| J04 | API de Hugging Face | `runs/p3/rebel/revision.json` |
| J05 | SHA de J04 | `runs/p3/rebel/snapshot.json` |
| J06 | ningun documento | `runs/p3/rebel/native_smoke_status.json` con `BLOCKED_FIXTURE` |
| J07 | `documents.jsonl` canonico, 36 lineas, diagnostico | `runs/p3/rebel/hohfeld-cross-lingual/` |
| J08 | `universal-ie/UIE@c88dd08b8ad0016ec597ebc6f8fff49480367bb2`, `btaille/sincere@dd1c34916ddcdc5ceb2799d64b17e80cdf1a5b31` | `upstreams/UIE`, `upstreams/sincere` |
| J09 | clones mas manifiesto versionado | datos reconstruidos bajo el clone en SCRATCH, `runs/p3/uie/rebuild_validation.json`, `shot_dir.txt` |
| J10 | test y train 10-shot | `runs/p3/uie/smoke_example.json` |
| J11 | `luyaojie/uie-base-en@966f8b1fc4c74e94ab552081605913ad5133cc41` | `runs/p3/uie/base_snapshot.json` |
| J12 | base pineada sin abrir el pickle, 10-shot, derivado safetensors aun no producido | `runs/p3/uie/preflight/jobs/<SLURM_JOB_ID>/preflight_report.json` |
| J13 | fingerprint identificado y autorizacion explicita, hoy cerrada | `runs/p3/uie/train/seed1-10shot/`, `training_manifest.json` |
| J14 | best checkpoint y ejemplo J10 | `runs/p3/uie/positive_smoke/` |
| J15 | L40S, sin pesos 7B, sin compilacion | `runs/p3/gollie/preflight/jobs/<SLURM_JOB_ID>/preflight_report.json` |
| J16 | `HiTZ/GoLLIE-7B@d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9` | `runs/p3/gollie/snapshot.json` |
| J17 | fixtures `tests/fixtures/gollie/official_re_*` | `runs/p3/gollie/native-smoke/` |
| J18 | Zenodo `genie_r.ckpt`, 4879373206 bytes, MD5 `c214da56b6e5d5bd259e0cbe826d92f7` | `models/genie/genie_r.ckpt` |
| J19 | `martinjosifoski/genie-rw@81eb2ccca714fbb62a65283b65c9c153469ee409` | `runs/p3/genie/tokenizer_snapshot.json` |
| J20 | Zenodo `tries.zip`, 94255261 bytes, MD5 `45db4dc2578cdeda891d8ed6ddb3f89f` | `models/genie/tries.zip` |

mREBEL conserva `es_XX` / `tp_XX`, float32, `max_input_tokens=256`, `max_new_tokens=512`, `num_beams=3`, `seed=42`, ventanas TOKEN, overlap 0. Nunca pasa `--legacy-relation-filter`.

REBEL no usa tokens `es_XX` / `tp_XX`. El texto espanol de J07 no se traduce. `max_length=512`. J05 y J07 pasan `--local-files-only`. J06 no llama al CLI.

UIE de entrenamiento: epochs 200, batch 16, lr 1e-4, scheduler constant, warmup_ratio 0.06, spot_noise 0.1, asoc_noise 0.1, meta_negative -1, random prompt (`ordered_prompt=False`), schema de evaluacion ordenado completo, max_source 256, max_target 192, constraints desactivados, argumento de split `"1"`, seed del trainer 42, best metric `eval_overall-F1`. `PYTHONHASHSEED=0` solo en J09, J12 y J13. `eval_steps` lo calcula en el cluster `scripts/get_eval_batch_num.py`; no esta horneado en el modulo de configuracion. El dataset convertido no se copia fuera del workspace privado.

Un checkpoint local afinado no recibe `--model-revision`. El CLI rechaza esa combinacion. `resolved_revision` pasa a ser `local-weights:` mas el SHA-256 del contenido de `config.json` y de los ficheros de pesos. `base_model_id` y `base_revision` conservan `luyaojie/uie-base-en` y `966f8b1fc4c74e94ab552081605913ad5133cc41` como procedencia del modelo base, no como revision de los pesos afinados. El registro incluye los hashes de los pesos y la identidad del training run (`output_dir`, metrica, SHA del manifiesto y metadatos SLURM). No se inventa una revision Git para esos pesos. Una carga hub de `uie-base-en` sigue resolviendo el SHA Git.

El fingerprint incluye python, torch, CUDA de torch, transformers, sentencepiece, nombre del device CUDA, commit upstream de UIE, revision del checkpoint base, hashes de scripts y config, los cuatro hashes del 10-shot frente al manifiesto, y el SHA-256 del `model.safetensors` derivado. No incluye timestamp, hostname ni UUID de GPU. No sustituye el `.bin` de origen por ese derivado: la conversion que produciria el safetensors esta deshabilitada.

## Criterios PASS y FAIL

PASS de un job es exit 0 mas el artefacto indicado arriba. FAIL es exit distinto de cero. No hay fallback a otra GPU.

El smoke positivo de UIE puede salir 0 con `relation_recovery=not_observed` si la estructura Spot-Asoc parseada no esta vacia. Eso es validez tecnica del loader y del parser, no una metrica de extraccion.

J14 llama al CLI limpio con el directorio local del checkpoint y `--uie-training-manifest`. No pasa `--model-revision`. Si Transformers no puede cargar ese directorio, J14 falla. Si el manifiesto resultante presenta el SHA de `uie-base-en` como revision de los pesos, J14 tambien falla.

## Dry-run y submit

Desde el checkout, sin enviar nada:

```bash
bash scripts/cesvima/submit_train_first.sh --dry-run --mrebel
bash scripts/cesvima/submit_train_first.sh --dry-run --all
bash scripts/cesvima/submit_train_first.sh --dry-run --rebel-cross-lingual
```

Sin una rama (`--mrebel`, `--rebel`, `--rebel-cross-lingual`, `--uie`, `--gollie`, `--genie-assets` o `--all`) el submitter sale 2 y no llama a `sbatch`. `--all` envia los jobs READY de reproduccion nativa y de Hohfeld espanol de esta fase: J00-J05, J08-J20 de UIE, GoLLIE y los tres activos de GenIE. No envia J06, J07, Pythia, inferencia GenIE ni Priority B.

`--rebel` envia J00, J04 y J05. J06 queda fuera hasta que exista un fixture ingles oficial. `--rebel-cross-lingual` anade J07 detras de J05.

`--dry-run` imprime el DAG y los comandos, asigna ids sinteticos `DRY_J00` y escribe `--dependency=afterok:DRY_J00` cuando corresponde. No crea el directorio de logs y no ejecuta `sbatch`.

El envio real solo se acepta si el checkout desde el que se lanza es `SYSTEM_ROOT` en CESVIMA. Captura el JOBID `--parsable` y encadena `afterok`. Si falta un script o un sbatch de la rama elegida, sale antes de enviar cualquier job de esa invocacion. No envia Pythia ni inferencia GenIE.

No se ha documentado una cuenta SLURM ni un mail para este usuario. Los jobs no llevan `--account` ni `--mail-user`.

## Que queda bloqueado

- J06 REBEL nativo: `BLOCKED_FIXTURE`. Priority A se mantiene. Falta un documento ingles oficial versionado. No se inventa.
- Pythia: `TARGET / BLOCKED_LICENSE`. No hay job. No se descarga el checkpoint fine-tuned. Sigue en Priority A.
- GenIE inferencia: `TARGET / BLOCKED_LOADER`. El extractor actual lanza `RuntimeError` antes de generar, tambien con `--authorize-checkpoint-load`. El estado de carga modernizada sigue `unverified`. No se inventa un loader Lightning. J18-J20 solo adquieren y verifican. La resolucion del loader queda para una iteracion posterior.
- Priority B (KnowPrompt, SynthIE, InstructUIE, KeyCARE): `STANDBY`. Sin jobs.
- Red de salida en nodos de computo: no verificada. Los jobs que necesitan Hugging Face, GitHub o Zenodo fallan cerrados. No hay un camino alternativo inventado por el nodo de login.
- CoNLL04 `public_redistribution` sigue `UNRESOLVED`. El uso privado en CESVIMA no autoriza a publicar el convertido.

## Frontera con C0/C1

C0 y C1 no estan en este DAG. Siguen sin backbone comun controlado, sin prompt comun, sin gramatica de salida, sin payload de ejecucion C0 distinto de `documents.jsonl`, y sin payload ni extractor C1. Las etiquetas C1 existen en `re_te_benchmark/schemas/c1/hohfeld_labels.json` (`Duty`, `Right`, `Privilege`, `NoRight`) y no se usan aqui. `CONDITIONING_SPEC` congela metadatos, no prompts. C2 sigue parcial y C3 no esta listo. Esta capa no los implementa.

## Checkpoint operativo 2026-10-26

Lo observado el 26 de octubre, con jobs, la recuperacion de bytes UIE tras el fallo de J09 `1348833` y el smoke tecnico Qwen3, esta en `docs/CESVIMA_CHECKPOINT_2026-10-26.md`. Ese fichero no se reescribe. Los gates de este documento son el contrato vigente e incorporan la recuperacion EXP.4.5. J13, J16, J17 y Gold C0/C1 siguen sin autorizar. J21 no entra en `submit_train_first.sh`.

La recuperacion local posterior a ese checkpoint, con el pin del `.bin`, la extension FlashAttention anterior al wheel corregido y el procedimiento de sincronizacion que no se ejecuta desde aqui, esta en `docs/CESVIMA_EXP45_RECOVERY.md`. EXP.4.5.1 deja escrita la secuencia futura de ABI y kernel para GoLLIE, y la de conversion y J12 para UIE. Ninguna de esas fases esta activa. Ese procedimiento no autoriza J13, J16, J17 ni Gold C0/C1.

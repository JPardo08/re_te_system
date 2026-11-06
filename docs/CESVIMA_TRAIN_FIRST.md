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
- J12 no abre `pytorch_model.bin`. El pin de origen es SHA-256 `04baf49c28f4be89424f3a61dd30d674c4ab165ff16c1b007525829a8c9441f7` y 990285657 bytes, revision `966f8b1fc4c74e94ab552081605913ad5133cc41`; coincidir con ese pin no autoriza la deserializacion. La conversion del checkpoint real es un job CPU aparte, `jobs/cesvima/uie_safetensors_convert.sbatch`, particion `standard`, 4 CPU, 32G y 02:00:00, sin GPU. El hash de tensores es un solo hilo: mas CPU o una GPU no lo acortan. 32G cubre el estado cargado y su clon; el job `1362612` llego a 2706560K de RSS y gasto la ventana en CPU antes de escribir `model.safetensors`. El limite de dos horas se mantiene como techo, no como remedio. No entra en `submit_train_first.sh --all`. Solo arranca si el entorno trae `UIE_SAFETENSORS_CONVERT_AUTHORIZED` igual a `1` y `UIE_CONVERT_PYTHON` es un ejecutable con PyTorch igual o posterior a 2.6 cuyo path no contiene `uie-train`. `env.sh` no define esa variable. La unica lectura del pickle es `torch.load(..., map_location="cpu", weights_only=True)`, y `_load_origin_bin` repite la autorizacion, el suelo de PyTorch 2.6 y la independencia de `uie-train` antes de esa llamada. Un tensor que no quede en CPU aborta. El archivo `pytorch_model.bin` se hashea antes y despues de la lectura; si su tamano o su SHA-256 cambian, la conversion aborta. No se rehasea el directorio entero. El derivado se escribe fuera de la cache de Hugging Face, en `models/uie/luyaojie-uie-base-en-966f8b1fc4c74e94ab552081605913ad5133cc41-safetensors/`. El snapshot J11 observado contiene `pytorch_model.bin`, `config.json`, `tokenizer_config.json`, `spiece.model`, `tokenizer.json`, `special_tokens_map.json`, `added_tokens.json`, `README.md` y `.gitattributes`. Son enlaces de la cache de Hugging Face; el destino legitimo puede estar en `blobs/` de `models--luyaojie--uie-base-en`. Un enlace se acepta solo si el destino real es un archivo dentro de SCRATCH y dentro de ese repositorio de cache. Un enlace que salga de la cache o de SCRATCH aborta. Los acompanantes se copian como archivos regulares, no como enlaces. Obligatorios solo `config.json` y `tokenizer_config.json`; el resto se copia si esta en el snapshot. No se copia ningun `.bin`. El informe guarda el SHA-256 de cada acompanante copiado. Antes de guardar registra los almacenamientos compartidos completos, clona esos tensores sin cambiar sus valores y vuelve a cargar el Safetensors para comparar claves, formas, dtypes y el SHA-256 de los bytes contiguos en CPU, little-endian, sin cabecera. El hash no incluye dtype ni shape: van en campos aparte. Esos bytes se leen con una vista uint8 del tensor CPU contiguo, no construyendo un `bytes` de Python elemento a elemento. El job imprime una linea `STAGE` al terminar cada fase, con flush y tiempo monotonico. Si la linea siguiente no aparece, esa fase sigue en curso. Si el alias fisico desaparece al clonar, el informe lo dice; no se finge que el archivo derivado conserve el alias. Una vista parcial que comparta almacenamiento aborta. Antes de publicar se comprueba el espacio libre del sistema de archivos. Si `quota -u -w` devuelve una sola fila con limite positivo, ese restante se interpreta como bloques de 1 KiB y tambien debe cubrir el doble del `.bin`; una salida ambigua o ausente deja la cuota individual como no observada y no se sustituye por el espacio global. El parcial es privado del JOBID. El directorio final no se sustituye si ya existe. El informe `CONVERSION_PASS` se crea con `O_EXCL` y solo despues el puntero `uie/conversion/accepted_job_id.txt`, tambien con `O_EXCL`. Un `publish.lock` exclusivo serializa esa publicacion y solo lo retira el JOBID que lo creo. Un fallo deja `CONVERSION_FAIL` en `uie/conversion/jobs/<SLURM_JOB_ID>/` y no escribe el puntero. Un directorio materializado sin ese puntero no esta aceptado. No se reescribe un informe ya existente ni se borra un artefacto de otro JOBID, ni un lock ajeno. Si el proceso muere con el lock puesto y sin puntero, el siguiente job falla cerrado; quitar ese lock es una recuperacion manual, no automatica. J12 consume solo ese puntero y ese informe. Recalcula el SHA-256 de `model.safetensors`, vuelve a cargar ese Safetensors y compara claves, formas, dtypes y hashes de valores con el informe, y rehasea `config.json`, `tokenizer_config.json` y el resto de acompanantes declarados. Un JSON coherente consigo mismo no basta si no coincide con esos bytes. No abre el pickle de origen, asi que los registros de tensores de origen siguen siendo la evidencia del job de conversion. Rechaza cualquier `.bin` en el directorio derivado y llama a `from_pretrained` con `use_safetensors=True` y `local_files_only=True`. Esa carga exige ademas `UIE_J12_DERIVED_LOAD_AUTHORIZED` igual a `1`. El sbatch de J12 no asigna ese valor: lo conserva solo si el operador lo exporta, y lo retira si llega vacio. La autorizacion de conversion no concede la de J12. Sin el informe aceptado o sin esa variable, J12 escribe `FAIL` en `uie/preflight/jobs/<SLURM_JOB_ID>/` antes de cargar pesos. No reescribe `uie/preflight/preflight_report.json` ni el informe de otro JOBID, incluido el historico `1349123` si ese directorio ya existe. No parchea el trainer ni la puerta de seguridad de Transformers. `deserialization_authorized()` sigue en false y las constantes `CONVERSION_ENABLED`, `UIE_J12_DERIVED_LOAD_AUTHORIZED` y `UIE_J13_TRAINING_AUTHORIZED` siguen en false.
- J13 no arranca las 200 epocas en esta iteracion. La constante `UIE_J13_TRAINING_AUTHORIZED` permanece false y no concede el entrenamiento: si deja de ser false, el job tambien se niega. La autorizacion de una sola ejecucion es `UIE_J13_RUN_AUTHORIZED` igual a `1` en el entorno de ese proceso. No autoriza J12, la conversion ni el micropreflight, y el sbatch no le asigna valor: lo conserva solo si el operador lo exporta. Hace falta ademas el puntero de J12 con JOBID `1363190`, su informe `PREFLIGHT_PASS`, la conversion `1363171`, el fingerprint `b585f1d9e2fa81cbd6503d47dbb7da64fbf63acf79cbd4a88725843ac6884530`, `model.safetensors` de 1187420216 bytes y SHA-256 `2623672bf5b80a1e9b78f0dd5ccc609792255e58c8845b556ab5822920f3bfd5`, la configuracion congelada, un directorio de salida vacio, un `publish.lock` libre y un informe `PROBE_PASS` del micropreflight. Esos JOBID, el fingerprint y el hash del derivado son evidencia acreditada en CESVIMA; este checkout no los ha vuelto a medir. El loss `4.218808` de J12, en 99 segundos y sobre una Tesla V100-PCIE-16GB, no es una metrica de evaluacion. El trainer pineado, commit local `_legacy/UIE` `c88dd08b8ad0016ec597ebc6f8fff49480367bb2`, asigna `checkpoint = None` despues de la rama `from_checkpoint`, asi que `resume_from_checkpoint` no reanuda. Un directorio de salida con ficheros se rechaza. Transformers 4.57.6 no tiene el campo `evaluation_strategy`: con el argv congelado la evaluacion queda en `no` y `load_best_model_at_end` aborta. El lanzador `uie_trainer_launch.py` traduce ese flag a `--eval_strategy=steps` sin cambiar el valor. El mismo argv no fija `save_steps`; el default 500 de la libreria no es multiplo del `eval_steps` calculado y 4.57.6 tambien aborta. El lanzador iguala `save_steps` a ese `eval_steps`. No cambia batch 16, eval batch 64, LR `1e-4`, 200 epocas, scheduler, warmup, semilla 42 ni la metrica `eval_overall-F1`. La cadencia de guardado pasa a coincidir con la evaluacion; `save_total_limit` sigue en 1. `training_step` de 4.57.6 recibe `num_items_in_batch` y `_maybe_log_save_evaluate` recibe siete argumentos; el trainer pineado no los acepta. El lanzador sustituye esos dos metodos y repone `_max_length`, `_num_beams` y `use_amp`, que la libreria ya no define. `_max_length` queda vacio al construir el trainer porque `generation_max_length` no esta en el argv congelado. La evaluacion automatica llama a `evaluate()` sin argumentos y el `prediction_step` pineado usa `config.max_length`, que `run_uie_finetune.py` iguala a `max_target_length` antes de cargar el modelo. En el job `1364429` esa evaluacion calculo `eval_overall-F1`. Al guardar, Transformers 4.57.6 mueve ese `max_length` a `generation_config` y deja `config.max_length` en `None`. La evaluacion explicita posterior llama a `evaluate(max_length=val_max_target_length, num_beams=num_beams)`. `val_max_target_length` no esta en el argv y el upstream lo copia de `max_target_length`. `num_beams` tampoco esta y su default es `None`. Transformers 4.57.6 guarda esos argumentos en `_gen_kwargs` y ya no asigna `_max_length`. El metodo pineado no lee `_gen_kwargs`, vuelve a la config vacia y `generate()` recibe `max_length=None`. `max_new_tokens` no forma parte del argv ni de esa llamada. El lanzador resuelve la longitud antes del metodo pineado: un entero explicito de `evaluate` o `predict` manda; si no hay, usa `config.max_length` y despues `generation_config.max_length`. Un `num_beams` explicito `None` no se reenvia. La funcion restringida del decoder sigue construyendose en el metodo pineado. La carga del hijo obliga `use_safetensors=True` y `local_files_only=True`, retira el argumento `mirror` del upstream y no toca la puerta de `torch.load`. `datasets` 5 ya no ejecuta `uie_json.py`; el lanzador lee los JSONL con el esquema pyarrow de `etc/record.dataload.schema`. El upstream no se edita, para que el fingerprint de J12 siga siendo comparable. El micropreflight `jobs/cesvima/j13_uie_gpu_probe.sbatch` pide la misma V100, 8 CPU y 32G, con una hora de techo. No entra en `--all` y no llama a J13 ni al entrenamiento upstream. Su sbatch retira la autorizacion de entrenamiento. El operador acredita el job `1363874` como `PROBE_PASS` en una Tesla V100-PCIE-16GB: forward y backward de batch 16 con pico `max_memory_allocated` de 14275008512 bytes, generacion de batch 64 con pico 3223645696 bytes y un checkpoint de prueba de 990185320 bytes. Este checkout no ha vuelto a medir esos bytes. Cada fase llama a `reset_peak_memory_stats` antes de trabajar, y `max_vram_bytes` se lee despues de ese reinicio en la fase de generacion, asi que ese campo es el pico desde la generacion y queda por debajo del pico de batch 16. El probe no registra memoria reservada ni libre, no llama a `cuda.synchronize` y no llama a `optimizer.step`; el pico de 14275008512 bytes no incluye el estado de Adam. `torch.optim.AdamW` tiene `weight_decay` 0.01 por defecto, pero al no haber `step` ese valor no actualizo pesos. El derivado aceptado mide 1187420216 bytes y el checkpoint de prueba 990185320; la diferencia es 197234896 bytes. La conversion clona los almacenamientos compartidos antes de escribir, y `save_pretrained` puede omitir nombres atados. Esos tamanos no identifican los tensores ni prueban valores iguales. Un `PROBE_PASS` no abre las 200 epocas. J13 sigue cerrado. La copia local de `seed1/10shot` coincide con los cuatro SHA-256 congelados: 50 registros JSON en train, 231 en validacion y 288 en test. `scripts/get_eval_batch_num.py` sobre ese `train.json` imprime 80. Con batch 16 y 200 epocas son 4 pasos por epoca, 800 pasos de optimizacion y 10 evaluaciones programadas, mas el `do_predict` final. Esos bytes son los del hash congelado, no una relectura del arbol de CESVIMA. Construir `TrainingArguments` en Transformers 4.57.6 exige `accelerate>=0.26.0`. El bootstrap lo instala solo al crear `uie-train`; un venv que ya existe no se modifica. La metrica importa `nltk.tree.ParentedTree`. El job `1364429` calculo la metrica, asi que ese venv ya tiene `nltk`. `click`, `joblib`, `cloudpickle` y `defusedxml` no los importa el trainer pineado. Este checkout no registra sus versiones en CESVIMA y el bootstrap no los anade: un venv existente sale antes de `pip` y no se sustituye ningun paquete acreditado. Si falta, J13 y el micropreflight se niegan antes de cargar pesos. `resize_token_embeddings` sigue dentro del trainer cuando faltan piezas especiales y no se ha medido sobre el derivado.
- J14 PASS tecnico si `parsed_structures` no esta vacio y la procedencia del checkpoint no usa el SHA base como revision de los pesos. La recuperacion de la relacion gold se informa aparte y no decide ella sola el exit code. Si la estructura esta vacia, o si `resolved_revision` es `966f8b1fc4c74e94ab552081605913ad5133cc41`, el exit es distinto de cero.
- J15 no alcanza `PREFLIGHT_PASS` mientras la identidad binaria no sea `MATCH`. Un `BINARY_MISMATCH` deja `abi_pass` en false y `abi_verdict` en `not_decided`. El informe va a `gollie/preflight/jobs/<SLURM_JOB_ID>/` e incluye el JOBID y el SHA-256 del wheel. El JSON antiguo `gollie/preflight_report.json` no se reescribe y no autoriza nada. Siguen vigentes la L40S, `torch==2.5.1+cu121`, el matmul bfloat16, Black `>=24.8,<25`, el kernel minimo y el import de `modeling_flash_llama.py`, pero solo despues de acreditar el binario. Esos 40 GiB libres siguen siendo un umbral operativo: el unico dato de memoria sustanciado en el repo es 13477.10 MB de notebook, no un tamano hasheado del checkpoint 7B.
- J16 no descarga si el puntero de JOBID falta, si el informe es de otro JOBID, si el estado no es `PREFLIGHT_PASS`, si la identidad binaria no es `MATCH`, si faltan `BUILD_PASS` o `ABI_PASS`, o si `weights_downloaded` no es false. El booleano `GOLLIE_J16_DOWNLOAD_AUTHORIZED` del modulo permanece false. La descarga solo arranca si el entorno del job trae esa variable igual a `1`, el interprete es `gollie-isolated` y no `gollie-l40s`, hay al menos 40 GiB libres en SCRATCH, `HF_HUB_CACHE` es escribible dentro de SCRATCH y la metadata del repositorio responde sin 401 ni 403. Si el directorio local existe pero le faltan archivos, J16 no usa `cache_or_download`: ese helper devolveria el directorio parcial sin completarlo. Pide el snapshot con `local_files_only` desactivado y solo entonces exige `config.json`, `modeling_flash_llama.py` de 46523 bytes, el indice si existe y todos los pesos que referencia. Un resultado que sigue incompleto no escribe `DOWNLOAD_PASS`. El informe queda en `gollie/download/jobs/<SLURM_JOB_ID>/`. No usa `--all`.
- J17 sigue bloqueado salvo `GOLLIE_J17_SMOKE_AUTHORIZED=1` en el entorno del job, valor que `env.sh` no define, un informe `DOWNLOAD_PASS` de J16 y un informe `CONVERSION_PASS` para `HiTZ/GoLLIE-7B@d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9`. La conversion es un job CPU aparte, con un interprete que no es `gollie-isolated` ni `gollie-l40s` y con PyTorch >=2.6. Lee solo los shards de `pytorch_model.bin.index.json` mediante `torch.load(..., map_location="cpu", weights_only=True)` y escribe Safetensors en `P3_MODELS_ROOT`, sin tocar la cache ni los `.bin`. Esos dos shards suman 13477726551 bytes. El smoke carga el derivado con `use_safetensors=True` y no vuelve a los `.bin`. El sbatch de inferencia apunta `GOLLIE_VENV` a `gollie-isolated` despues de cargar `env.sh`. La instancia cargada declara que no admite DynamicCache automatico mediante `_supports_default_dynamic_cache`. Ese registro queda en `operational_metadata.cache_compatibility` y no entra en el calculo del `run_id`. Durante `generate()` eager, si `torch.library.wrap_triton` no existe, un contexto lo define como la identidad del kernel Triton `rotary_kernel` y lo retira en `finally`. Ese segundo registro queda en `operational_metadata.rope_compatibility` y tampoco entra en el `run_id`. Cada job escribe en `gollie/native-smoke/jobs/<SLURM_JOB_ID>/`, con su propio stdout, y no reescribe ejecuciones anteriores. El criterio tecnico sigue siendo cuatro artefactos, `inputs_succeeded==1` y esa revision. Una generacion vacia no es, por si sola, el criterio de PASS. El stdout del CLI solo vale si su ultima linea no vacia es un unico objeto JSON con `run_dir`; las lineas anteriores se conservan en el log y una salida ambigua o malformada falla. `jobs/cesvima/j17_gollie_trace.sbatch` no entra en `submit_train_first.sh`. Solo si el entorno trae la traza autorizada, una excepcion de extraccion deja `extraction_traceback.txt` y el traceback en stderr, sin anadir campos a `failures.jsonl` ni relajar ese criterio.
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
| UIE conversion | `pytorch_model.bin` pineado; job CPU fuera de `--all` | `runs/p3/uie/conversion/jobs/<SLURM_JOB_ID>/conversion_report.json`; aceptado solo si `accepted_job_id.txt` nombra ese JOBID tras `CONVERSION_PASS` |
| J12 | informe de conversion aceptado y 10-shot congelado; la carga sigue cerrada sin la variable de J12 | `runs/p3/uie/preflight/jobs/<SLURM_JOB_ID>/preflight_report.json` |
| J13 | fingerprint identificado y autorizacion explicita, hoy cerrada | `runs/p3/uie/train/seed1-10shot/`, `training_manifest.json` |
| J14 | best checkpoint y ejemplo J10 | `runs/p3/uie/positive_smoke/` |
| J15 | L40S, sin pesos 7B, sin compilacion | `runs/p3/gollie/preflight/jobs/<SLURM_JOB_ID>/preflight_report.json` |
| J16 | `HiTZ/GoLLIE-7B@d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9` | `runs/p3/gollie/snapshot.json` |
| J17 | fixtures `tests/fixtures/gollie/official_re_*` | `runs/p3/gollie/native-smoke/jobs/<SLURM_JOB_ID>/` |
| J18 | Zenodo `genie_r.ckpt`, 4879373206 bytes, MD5 `c214da56b6e5d5bd259e0cbe826d92f7` | `models/genie/genie_r.ckpt` |
| J19 | `martinjosifoski/genie-rw@81eb2ccca714fbb62a65283b65c9c153469ee409` | `runs/p3/genie/tokenizer_snapshot.json` |
| J20 | Zenodo `tries.zip`, 94255261 bytes, MD5 `45db4dc2578cdeda891d8ed6ddb3f89f` | `models/genie/tries.zip` |

mREBEL conserva `es_XX` / `tp_XX`, float32, `max_input_tokens=256`, `max_new_tokens=512`, `num_beams=3`, `seed=42`, ventanas TOKEN, overlap 0. Nunca pasa `--legacy-relation-filter`.

REBEL no usa tokens `es_XX` / `tp_XX`. El texto espanol de J07 no se traduce. `max_length=512`. J05 y J07 pasan `--local-files-only`. J06 no llama al CLI.

UIE de entrenamiento: epochs 200, batch 16, lr 1e-4, scheduler constant, warmup_ratio 0.06, spot_noise 0.1, asoc_noise 0.1, meta_negative -1, random prompt (`ordered_prompt=False`), schema de evaluacion ordenado completo, max_source 256, max_target 192, constraints desactivados, argumento de split `"1"`, seed del trainer 42, best metric `eval_overall-F1`. `PYTHONHASHSEED=0` solo en J09, J12 y J13. `eval_steps` lo calcula en el cluster `scripts/get_eval_batch_num.py`; no esta horneado en el modulo de configuracion. El dataset convertido no se copia fuera del workspace privado.

Un checkpoint local afinado no recibe `--model-revision`. El CLI rechaza esa combinacion. `resolved_revision` pasa a ser `local-weights:` mas el SHA-256 del contenido de `config.json` y de los ficheros de pesos. `base_model_id` y `base_revision` conservan `luyaojie/uie-base-en` y `966f8b1fc4c74e94ab552081605913ad5133cc41` como procedencia del modelo base, no como revision de los pesos afinados. El registro incluye los hashes de los pesos y la identidad del training run (`output_dir`, metrica, SHA del manifiesto y metadatos SLURM). No se inventa una revision Git para esos pesos. Una carga hub de `uie-base-en` sigue resolviendo el SHA Git.

El fingerprint incluye python, torch, CUDA de torch, transformers, sentencepiece, nombre del device CUDA, commit upstream de UIE, revision del checkpoint base, el JOBID de la conversion aceptada, el SHA-256 de la procedencia tensorial, hashes de scripts y config, los cuatro hashes del 10-shot frente al manifiesto, y el SHA-256 del `model.safetensors` derivado. No incluye timestamp, hostname ni UUID de GPU. No abre el `.bin` de origen. Antes de ese fingerprint, J12 vuelve a medir los tensores del Safetensors y los SHA-256 de los acompanantes declarados. El derivado real no existe hasta que el job CPU autorizado lo publique, y J12 no lo carga sin su propia variable.

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

Sin una rama (`--mrebel`, `--rebel`, `--rebel-cross-lingual`, `--uie`, `--gollie`, `--genie-assets` o `--all`) el submitter sale 2 y no llama a `sbatch`. `--all` envia los jobs READY de reproduccion nativa y de Hohfeld espanol de esta fase: J00-J05, J08-J20 de UIE, GoLLIE y los tres activos de GenIE. No envia J06, J07, el convertidor UIE, Pythia, inferencia GenIE ni Priority B.

La conversion UIE y una repeticion de J12 quedan fuera de ese DAG. No se lanzan desde este checkout. El interprete candidato ya comprobado en CESVIMA el 04/11/2026 es `/home/y064/SCRATCH/y064644/paper3/venvs/re_te_system_312/bin/python`: Python 3.12.3, Torch `2.14.0+cu130`, Safetensors 0.8.0, `torch.load` con `weights_only`, independiente de `uie-train`. Esa compilacion incluye CUDA; el sbatch no pide GPU, vacia `CUDA_VISIBLE_DEVICES` y la conversion exige tensores CPU. No es un valor global del repositorio: entra solo por `UIE_CONVERT_PYTHON`. Con autorizacion explicita, en CESVIMA y con un JOBID nuevo, los comandos son:

```bash
sbatch --export=ALL,UIE_SAFETENSORS_CONVERT_AUTHORIZED=1,UIE_CONVERT_PYTHON=/home/y064/SCRATCH/y064644/paper3/venvs/re_te_system_312/bin/python jobs/cesvima/uie_safetensors_convert.sbatch
sbatch --export=ALL,UIE_J12_DERIVED_LOAD_AUTHORIZED=1 jobs/cesvima/j12_uie_preflight.sbatch
```

J13 no se lanza desde esta iteracion. Un PASS de J12 no lo enciende y el `PROBE_PASS` de `1363874` tampoco. El contrato de lineas de la validacion de integracion, en el nodo de login y con el Python de `uie-train`, es de solo lectura:

```bash
UIE_J13_INTEGRATION_PROBE_AUTHORIZED=1 "${UIE_TRAIN_VENV}/bin/python" scripts/cesvima/j13_uie_integration_probe.py --cpu-only
```

Ese comando no pide GPU, no escribe `INTEGRATION_PASS` y no autoriza J13. La validacion GPU, no enviada desde este checkout, usa un JOBID, un informe y un directorio propios:

```bash
sbatch --export=ALL,UIE_J13_INTEGRATION_PROBE_AUTHORIZED=1 jobs/cesvima/j13_uie_integration_probe.sbatch
```

Pide la misma V100, 8 CPU y 32G, con dos horas de techo. Escribe solo en `uie/j13-integration/jobs/<SLURM_JOB_ID>/`. Hace un paso real con AdamW, LR `1e-4` y `weight_decay` 0.0, que es el default que el argv congelado deja sin fijar, y despues un trainer limitado a `max_steps=1` y 16 ejemplos de train, validacion y test. Esas tres cifras no estan en `uie_train_config.py`. El `eval_steps` canonico de 80 y las 200 epocas siguen siendo el contrato de J13. El resultado no es cientifico. `INTEGRATION_PASS` deja `authorizes_j13` en false y no escribe el puntero de J13. Mientras exista `uie/j13-integration/publish.lock`, J13 se niega. Si el paso o el trainer agotan la memoria, el estado es `BLOCKED_RESOURCE_CONFIG` y no se cambian hiperparametros. El entrenamiento de 200 epocas, que no se ejecuta ahora, seguiria siendo un comando distinto y posterior:

```bash
sbatch --export=ALL,UIE_J13_RUN_AUTHORIZED=1 jobs/cesvima/j13_uie_train.sbatch
```

Ese comando no se ejecuta ahora. La variable no queda en `env.sh` y no abre otro job.

Esos comandos no se han ejecutado. Antes de sincronizar el checkout con CESVIMA hay que publicar el commit revisado y, en el cluster, actualizar solo los archivos de esta recuperacion. Alli estan modificados localmente `scripts/cesvima/env.sh` y `scripts/cesvima/j00_verify_infer_env.py`, ademas de auxiliares sin seguimiento. Esa sincronizacion no debe pisarlos. No se hace desde esta iteracion.

Despues de la conversion, y antes de conceder J12, se lee `uie/conversion/accepted_job_id.txt` y el `conversion_report.json` de ese JOBID. Hace falta `status=CONVERSION_PASS`, `verified=true`, `weights_only=true`, `origin_unchanged=true`, el pin de tamano y SHA-256, el derivado fuera de la cache y sin `.bin`, y los hashes de acompanantes. Un `CONVERSION_FAIL`, un directorio sin puntero o un lock sin puntero no autorizan la carga. El informe de J12 `1349123` no se reescribe.

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
- J13 entrenamiento: cerrado. El job `1363874` acredita que un forward y un backward de batch 16 con tokens aleatorios, y una generacion de batch 64, cupieron en la V100 de 16 GB. Ese pico no incluye `optimizer.step` ni la evaluacion del trainer. La V100, 8 CPU, 32G y 48 horas del sbatch de entrenamiento siguen siendo un presupuesto, no una medicion de las 200 epocas. Ni `PROBE_PASS` ni un futuro `INTEGRATION_PASS` autorizan J13.

## Frontera con C0/C1

C0 y C1 no estan en el DAG de `submit_train_first.sh`. El shell comun versionado es `common-instruction-shell-v2`, definido en `re_te_benchmark.conditioning.shell` y consumido por el harness de experimento controlado. Los slots C0 y C1 tienen payload: C0 esta vacio y C1 contiene el texto corto de etiquetas. Preparar ese payload no autoriza la ejecucion. J21 es un smoke de textos inventados en un arbol aislado; un pase tecnico de ese smoke no habilita inferencia sobre Gold ni sobre los 36 documentos. La revision observada de Qwen3 no queda pineada por el registro. C2 continua `PARTIAL`. C3 continua `NOT_READY` y `NOT_EXECUTABLE`. C3-NL sigue siendo una variante separada y no autoriza C3.

## Componentes auxiliares ya versionados

El DAG principal no cambia. `submit_train_first.sh --gollie` sigue enviando `jobs/cesvima/j15_gollie_preflight.sbatch`, despues J16 y J17. El J15 aislado, `jobs/cesvima/j15_gollie_isolated_preflight.sbatch`, coexiste y no sustituye ese envio. Cambiar el fichero que el submitter enviaria queda fuera de esta reconciliacion y necesita su propia validacion.

Tambien estan versionados, y fuera de `--all`:

- `jobs/cesvima/uie_safetensors_convert.sbatch`
- `jobs/cesvima/j13_uie_gpu_probe.sbatch`
- `jobs/cesvima/j13_uie_integration_probe.sbatch`
- `jobs/cesvima/gollie_safetensors_convert.sbatch`
- `jobs/cesvima/j21_qwen3_invented_smoke.sbatch`

La conversion, el preflight J12, el microprobe GPU y el probe de integracion de UIE son componentes distintos. J13 sigue sin autorizacion. J14 sigue a la espera de J13. Los hiperparametros de las 200 epocas no cambian. El entrenamiento exige el derivado Safetensors y rechaza `pytorch_model.bin`. Una prueba de integracion no es el entrenamiento cientifico. Exigir su informe como dependencia positiva de J13 queda para una iteracion posterior, despues de leer el JSON del job.

Una consulta de contabilidad SLURM del 05/11/2026 observo `COMPLETED 0:0` en J21 `1349043`, GoLLIE J15 `1351016`, J16 `1351274`, J17 `1351511`, conversion UIE `1363171`, J12 `1363190`, probe GPU `1363874` e integracion `1364903`. J09 `1348833` figura como `FAILED 1:0` y una recuperacion posterior `1349096` como `COMPLETED 0:0`. Esos estados no se han vuelto a contrastar con los informes de SCRATCH en esta reconciliacion. `COMPLETED` no es un PASS cientifico ni `VALIDATED` del registro. El resultado historico de J17 no cierra la campana GoLLIE.

## Gemma 4 G00-G02

`google/gemma-4-12B-it` es el backbone instruction-tuned. `google/gemma-4-12B` queda rechazado. La revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` es el unico pin de identidad aprobado. Ese pin no acredita que los pesos locales existan ni que se hayan descargado. El runtime de inferencia pineado a Transformers 4.57.6 no tiene `Gemma4UnifiedForConditionalGeneration`. El venv aislado es `GEMMA_VENV`, bajo `SCRATCH_ROOT/venvs/gemma-isolated`, y no se crea en esta iteracion. La documentacion publicada de Transformers situa esa clase en 5.10.1. No se reutiliza `uie-train` para cargarla.

G00, G01 y G02 tienen sbatch propio y no entran en `submit_train_first.sh`. G00 es CPU y no descarga. Su PASS acredita solo que `GEMMA_VENV` importa `Gemma4UnifiedForConditionalGeneration` con Transformers igual o posterior a 5.10.1. No aprueba canales, EOS/PAD ni VRAM, y no autoriza G01, G02, G03 ni G04. La revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` es un pin de identidad del checkpoint instruction-tuned. Ese pin no descarga, no carga la GPU y no abre Gold: G01 sigue exigiendo `GEMMA_G01_ACQUIRE_AUTHORIZED=1`. La copia publicada es por bloques, sin hardlinks, y el marcador de aceptacion se escribe solo despues de verificar el destino. Un directorio parcial no se borra solo. G01 solo puede adquirir ese pin, no `main`, ni un tag movil, ni otra revision, y no sobrescribe un snapshot aceptado. G02 pide `gpu:l40s:1` en el sbatch. La ejecucion solo continua si `scontrol` muestra AllocTRES `gres/gpu:l40s=1` y el dispositivo CUDA visible es una L40S. Carga solo en local y, si llega a ejecutarse, usa una frase inventada. No se ejecuta aqui. El tamano bf16 de 11.95B parametros es una estimacion de 23900000000 bytes, no una medicion de VRAM. C0, C1 y el shell v2 no cambian.

## Gemma 4 G03-G05

G03 prepara el smoke pareado C0/C1 sobre dos frases inventadas, con una sola carga de pesos cuando el runtime lo permita. El PASS es tecnico: no autoriza G04 ni es un F1. Los artefactos son `manifest.json`, `predictions.jsonl`, `failures.jsonl` y `statistics.json`, mas el desglose por condicion que ya usa el puente de evaluacion.

G04 nombra la cohorte de 36 documentos de `teresia_hohfeld` sin abrirla. Rechaza anotaciones, exige revision aprobada, snapshot, runtime, preflight G02, smoke G03, decodificacion congelada e identica, autorizacion humana y directorio seguro. Una variable de entorno no sustituye esas comprobaciones. El phase lock sigue cerrado, asi que el sbatch es de CPU y no reserva GPU. No se lanza Gold.

G05 llama a `evaluate_predicates` y `evaluate_triplets` de P0, con `strict_inline` y `exact_multiset_v1`. Los tres estimandos siguen siendo `conservative_operational_f1`, `f1_conditional_on_completed_generation` y `paired_secondary_f1`. El contraste C1-C0 no se publica. El sbatch rechaza una ruta Hohfeld.

## Checkpoint operativo 2026-10-26

Lo observado el 26 de octubre, con jobs, la recuperacion de bytes UIE tras el fallo de J09 `1348833` y el smoke tecnico Qwen3, esta en `docs/CESVIMA_CHECKPOINT_2026-10-26.md`. Ese fichero no se reescribe. Los gates de este documento son el contrato vigente e incorporan la recuperacion EXP.4.5. J13, J16, J17 y Gold C0/C1 siguen sin autorizar. J21 no entra en `submit_train_first.sh`.

La recuperacion local posterior a ese checkpoint, con el pin del `.bin`, la extension FlashAttention anterior al wheel corregido y el procedimiento de sincronizacion que no se ejecuta desde aqui, esta en `docs/CESVIMA_EXP45_RECOVERY.md`. EXP.4.5.1 dejo escritas las fases de UIE; sus constantes siguen inactivas. EXP.4.6 anade el convertidor CPU y la lectura J12 del derivado, ambos cerrados salvo variable de entorno. J12, si llega a autorizarse, mide de nuevo los tensores del Safetensors y los acompanantes; no abre el pickle. No convierte el checkpoint real. La cuota individual de BeeGFS sigue sin una medicion verificada: el espacio global observado de 67 TB no es esa cuota. El trainer, el tokenizer de entrenamiento y `resize_token_embeddings` siguen sin ejecutarse sobre el derivado. El job `1363874` cargo el derivado para forward y backward con tokens aleatorios y no ejecuto `optimizer.step`. La validacion `j13_uie_integration_probe.sbatch` es la que debe medir el paso completo, la evaluacion y el guardado. La contabilidad del 05/11 observa un job de integracion completado; este checkout no ha leido su JSON y ese resultado no autoriza J13. Ese procedimiento tampoco autoriza J16, J17 ni Gold C0/C1.

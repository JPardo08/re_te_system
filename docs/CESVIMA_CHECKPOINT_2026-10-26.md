# Checkpoint CESVIMA 2026-10-26

Registro operativo del cierre de la jornada del 26 de octubre de 2026. Describe el ultimo estado observado de los jobs, la recuperacion de artefactos UIE y las comprobaciones que quedan abiertas. No es un resultado cientifico, no cierra la campana y no modifica el contrato de `docs/CESVIMA_TRAIN_FIRST.md`.

Este fichero se escribio desde el workspace local, sin conexion al cluster. Los estados `COMPLETED`, `FAILED` y `RUNNING` son el ultimo checkpoint de la sesion. Un job SLURM sigue en ejecucion despues de cerrar SSH. Cualquier estado posterior al de este registro esta sin observar y no se inventa aqui.

## Alcance y linea base local

Repositorio: `re_te_system`. Rama publicada de referencia: `feat/paper3-experimental-integration`, commit `a65fb076f11908ddad55182363d91097f1357fdb` (2026-10-21). Ese commit ya esta en `origin`. El trabajo de esta jornada esta en el arbol de trabajo y, a partir de este fichero, en la documentacion. No forma parte de `a65fb07`.

`re_te_benchmark` y `re_te_evaluation` no recibieron cambios de esta jornada. `re_te_ontology_conditioned` no esta en el workspace.

J21 no pertenece al DAG `J00`-`J20` de `scripts/cesvima/submit_train_first.sh`. Su sbatch y su script son ficheros nuevos del arbol local. El codigo que se ejecuto dentro del arbol aislado de CESVIMA no se ha vuelto a leer durante este cierre, asi que la identidad exacta de `qwen3.py` dentro de ese arbol aislado queda sin reconfirmar.

## Jobs observados

| Job | ID SLURM | Ultimo estado observado | Lectura de este cierre |
| --- | --- | --- | --- |
| J21 Qwen3 smoke inventado | `1349043` | `COMPLETED 0:0` | Pase tecnico. Gold cerrado. |
| J09 reconstruccion CoNLL04 | `1348833` | `FAILED` | Divergencia byte a byte. Sigue siendo un fallo. |
| Validacion de la recuperacion UIE | `1349096` | `COMPLETED 0:0` | Integridad de bytes recuperados. No rehabilita J09. |
| J10 ejemplo congelado | `1349121` | `COMPLETED 0:0` | Ejemplo `conll04-test-0` sobre el 10-shot recuperado. |
| J11 cache del checkpoint base | `1348832` | `COMPLETED 0:0` | `luyaojie/uie-base-en@966f8b1fc4c74e94ab552081605913ad5133cc41`. |
| J12 preflight UIE | `1349123` | `RUNNING` en V100 `r2n6` | Resultado definitivo desconocido. Sin PASS. |
| J13 entrenamiento 200 epocas | sin ID de esta jornada | no enviado como autorizado | `NO AUTORIZADO`. |
| J15 GoLLIE preflight | `1349032` | `RUNNING`, unos 94 min | Sin `GOLLIE_SOURCE_BUILD_PASS` ni `GOLLIE_WHEEL_ABI_PASS`. Sin PASS. |

La cadena antigua conserva dependencias `afterok` sobre J09. En el submitter versionado, J10 espera a J09 y J12 espera a J09 y a J11. El fallo de `1348833` deja insatisfechas esas aristas. `1349121` y `1349123` son envios recuperados, distintos de los dependientes que sigan apuntando a `1348833`. Los JOBID de esa cola antigua no estan en el workspace local y no se inventan.

## Qwen3 J21

Job `1349043`, `COMPLETED 0:0`. Smoke tecnico en una GPU L40S. El contrato local define cuatro ejecuciones: `invented-smoke-01` y `invented-smoke-02`, cada una en C0 y en C1, con una sola carga de pesos. La sesion registro que esas cuatro inferencias sobre documentos inventados terminaron con `technical_pass=true`.

Checkpoint candidato:

```text
Qwen/Qwen3-4B-Instruct-2507
revision cdbee75f17c01a7cc42f958dc650907174af0554
```

`gold_inference_authorized=false`. No hay resultado cientifico, no hay F1 y no hay autorizacion de los 36 documentos Hohfeld. El plan C0/C1 del codigo sigue con `real_execution_authorized=false` y exige revision humana despues de un pase tecnico.

Salida prevista por el sbatch local, si el job uso ese fichero:

```text
/home/y064/SCRATCH/y064644/paper3/runs/p3/qwen3/invented-smoke/1349043
```

El arbol aislado declarado en el sbatch es:

```text
/home/y064/SCRATCH/y064644/paper3/isolated/qwen3-j21/re_te_system
```

Ese arbol no es `SYSTEM_ROOT`. Este cierre no ha comprobado que el fichero aislado coincida con el arbol local posterior a la correccion del indice.

### Anomalia del indice de pesos

La anomalia queda registrada y su aprobacion sigue pendiente. `index_sha256_approved` permanece `false`. Observar el digest del indice no lo convierte en un pin aprobado.

Cifras medidas sobre la revision candidata y escritas en el codigo local:

| Cantidad | Bytes |
| --- | ---: |
| `model-00001-of-00003.safetensors` | 3957900840 |
| `model-00002-of-00003.safetensors` | 3987450520 |
| `model-00003-of-00003.safetensors` | 99630640 |
| Suma fisica de los tres shards | 8044982000 |
| Almacenamiento de tensores | 8044936192 |
| Cabeceras safetensors | 45808 |
| `metadata.total_size` declarado | 8045591552 |
| Declarado menos almacenamiento de tensores | 655360 |

SHA-256 de los shards, ya contrastados con la allowlist en CESVIMA y copiados al codigo local:

```text
model-00001-of-00003.safetensors 75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6
model-00002-of-00003.safetensors 0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1
model-00003-of-00003.safetensors 7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d
```

`metadata.total_size` cuenta bytes de almacenamiento de tensor, no el tamano de los ficheros. La suma fisica se explica por almacenamiento mas cabeceras (`8044936192 + 45808 = 8044982000`). El indice publicado queda 655360 bytes por encima del almacenamiento medido. El verificador local acepta solo esa discrepancia ya registrada cuando la identidad de shards coincide; cualquier otra declaracion de `total_size` sobre esa identidad falla. El SHA-256 del indice sigue sin aprobar. El pase tecnico del smoke no abre Gold.

## Recuperacion UIE frente a la reconstruccion J09

Hay dos procedimientos distintos. Solo uno de ellos produjo bytes aceptados el 26 de octubre, y ese procedimiento no es una reconstruccion.

### Reconstruccion exigida por J09

`jobs/cesvima/j09_uie_rebuild_data.sbatch` exporta `PYTHONHASHSEED=0`. `scripts/cesvima/j09_uie_rebuild_data.py` sale si esa variable no vale `0`. Despues:

1. Comprueba el `conll04.json` del clone sincere pineado contra el manifiesto.
2. Escribe los jsonlines intermedios con `sincere_processing.py`.
3. Ejecuta `uie_convert.py` y `sample_data_shot.py -seed 1` con `PYTHONHASHSEED=0`.
4. Compara cada SHA-256 con `artifacts/uie_conll04_10shot_seed1_manifest.json`.
5. Ejecuta `scripts/reproductions/uie_conll04/validate_reproduction.py`.
6. Exige el hash combinado congelado y los recuentos 50/231/288.
7. Escribe `shot_dir.txt` solo si todo lo anterior coincide.

El job original `1348833` termino `FAILED` por divergencia de hashes byte a byte. Un mismatch es un fallo cerrado: el contrato no acepta un arbol parecido. `1348833` no queda rehabilitado por los jobs posteriores.

La investigacion del convertidor en esta jornada atribuyo esa divergencia a `PYTHONHASHSEED`, que cambia el orden de etiquetas y de esquemas. El mecanismo ya esta descrito en `docs/UIE_CONLL04_REPRODUCTION.md`, seccion 8, y en el manifiesto:

- `Text2SpotAsoc.annonote_graph` construye las etiquetas spot/asoc con conjuntos de Python.
- `uie_convert.py` serializa `list(spot_labels)` y `list(asoc_labels)`.
- La generacion de esquemas tambien convierte conjuntos en listas.
- El upstream no fija `PYTHONHASHSEED`.
- El freeze canonico uso `PYTHONHASHSEED=0` de forma prospectiva, regenero el arbol dos veces y obtuvo el mismo inventario. Ese valor no es la semilla de split ni la semilla del trainer.
- El codigo upstream no se reordena despues de generar.

`PYTHONHASHSEED=0` en J09 era la condicion necesaria ya versionada. En `1348833` no basto para reproducir los bytes congelados. Este cierre no anade otro experimento causal y no elige un hash historico a posteriori. El interprete de la reconstruccion original fue Python 3.8; el de CESVIMA en este DAG es el Python 3.12 verificado por J00. Esa diferencia de interprete ya estaba escrita en `docs/CESVIMA_TRAIN_FIRST.md`. No se declara aqui como causa demostrada del fallo `1348833`.

### Recuperacion de los artefactos originales

La recuperacion copia y verifica bytes ya generados. No vuelve a ejecutar el convertidor.

Procedencia local, nombrada por el manifiesto y no versionada:

```text
_legacy/sincere/data/conll04.json
_legacy/UIE/dataset_processing/data/relation/conll04/
_legacy/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04/
_legacy/UIE/dataset_processing/converted_data/text2spotasoc/relation/conll04_shot/seed1/10shot/
```

Esos derivados siguen ignorados por git. No se publican y no se modifican sus splits, etiquetas, esquemas ni hashes.

La sesion registro 11/11 verificaciones SHA-256 correctas en el Mac y 10/10 correctas despues de la transferencia a CESVIMA. El inventario exacto de esas dos cuentas esta en `recovery_registry.json` del cluster. Ese JSON no esta en el workspace local, asi que este cierre no renombra el 11 ni el 10.

Comprobacion adicional de este cierre, hecha en local y en solo lectura contra el manifiesto versionado: 18/18 rutas presentes y con el SHA-256 esperado. Entran la fuente sincere, los tres jsonlines intermedios, la conversion completa, el 10-shot y los cuatro esquemas tanto en el directorio completo como en el 10-shot. Varias de esas rutas repiten el mismo contenido (validacion y test del 10-shot coinciden con la conversion completa). Este 18/18 no sustituye el recuento 11/11 de la sesion ni demuestra la transferencia.

Directorio remoto de la recuperacion:

```text
/home/y064/SCRATCH/y064644/paper3/isolated/uie-frozen-recovery-20261026
```

Job `1349096`, `COMPLETED 0:0`. La sesion registro seis controles PASS. Dos quedan identificados: el hash combinado y el SHA-256 completo del informe. Los otros cuatro nombres viven en el log de `1349096` y en `recovery_registry.json`. No se listan aqui de memoria.

`recovery_registry.json` guarda la procedencia y el registro del J09 fallido. `shot_dir.txt` apunta al `conll04_shot/seed1/10shot` recuperado, que es la ruta que leen J10 y J12. J10 recuperado `1349121` termino `COMPLETED 0:0` y congelo el ejemplo `conll04-test-0`. Ese identificador es el que escribe `j10_uie_freeze_smoke_example.py` cuando la primera linea de test cuyo texto no esta en train tiene indice 0. El texto del ejemplo no se copia a este repositorio.

J11 `1348832` termino `COMPLETED 0:0` sobre el checkpoint base `luyaojie/uie-base-en`, revision `966f8b1fc4c74e94ab552081605913ad5133cc41`.

## Datos recuperados y conversion reproducida

| | Bytes recuperados | Conversion reproducida por J09 |
| --- | --- | --- |
| Que demuestra | Los ficheros locales y los transferidos coinciden con el freeze ya versionado. | El cluster regenera esos bytes desde los clones pineados y el manifiesto. |
| Estado el 2026-10-26 | Integridad verificada en la sesion (11/11 en Mac, 10/10 tras la copia) y, en este cierre, 18/18 rutas locales del manifiesto. | `1348833` `FAILED`. La reproduccion en CESVIMA no esta establecida. |
| Efecto sobre el DAG | `shot_dir.txt` puede alimentar J10 y J12 con el 10-shot congelado. | La arista `afterok` de `1348833` sigue insatisfecha. |
| Que no demuestra | Que el convertidor del cluster sea determinista. | Un corpus nuevo ni una correccion de solapes. |

El manifiesto versionado sigue siendo la autoridad de los hashes. No se ha editado. Su `cesvima_status` continua en `NOT_PREFLIGHTED`, `training_status` en `NOT_STARTED` y el bloque `checkpoint` en `NOT_STARTED`, porque J12 no tiene resultado y J13 no se ha autorizado. Cambiar esos campos ahora declararia un preflight que este cierre no ha visto.

### Avisos de solapamiento que se conservan

Salen de `artifacts/uie_conll04_10shot_seed1_manifest.json`, campo `split.leakage_warning`, con politica `record_without_manual_correction`. La seccion 6 de `docs/UIE_CONLL04_REPRODUCTION.md` los explica. Este cierre no los recalcula ni los corrige.

```text
full train/validation text overlap: 12
full train/test text overlap: 18
full validation/test text overlap: 3
shot train/validation text overlap: 0
shot train/test text overlap: 1
```

El mismo informe versionado conserva, sin correccion, textos duplicados dentro de train (24), validacion (3), test (1) y train del shot (2). El origen esta en el split upstream pineado. Quitarlos o reasignarlos romperia la regla de no correccion manual. Cualquier resultado cientifico futuro tiene que informarlos.

`public_redistribution` sigue `UNRESOLVED`. La copia privada a SCRATCH no autoriza a publicar el convertido.

### Hashes de autoridad ya versionados

Hash combinado del informe:

```text
8e813869fc6c0546b321e1dc8625f19ac17bced8e644b36ea117389f22542c91
```

SHA-256 del fichero `artifacts/uie_conll04_10shot_seed1_validation.json`, recomprobado en este cierre:

```text
efc6a7c106c8660177c16c4c61aec20a4dc678f0f2d61b89753074b54e8f21f8
```

Arbol generado, inventario de las dos pasadas locales historicas:

```text
5bc9929f808ac9acee82a3e71acf21302d3e55e43f774680d2ed5c232a88df75
e7e9b56b16e95afacf2686fdc4c132320355cf1798013a380de19be2fc0a845b
```

10-shot que J13 podria consumir solo despues de una autorizacion distinta de este cierre:

```text
train.json 5fecc7091bf7152d04a4eb43641933600f24f3af11995bc3e8f8d268fd43da7b
val.json   a9be8c5c3716456fb3caffa13f069feea6272a07a4fadce47265db07b30debb8
test.json  91d6b666268bacf73ea9352f7111c2ef84a212374148bb357933555707811feb
```

Fuente sincere `conll04.json`:

```text
3abe11eb5750e2c37e99a428f5e6c7e3b15f77d66053bca85e21e0772d601119
```

El resto de hashes de intermedios, conversion completa y esquemas esta en el manifiesto y en la seccion 8 de `docs/UIE_CONLL04_REPRODUCTION.md`. No se duplican otra vez.

## J12 y GoLLIE, aun abiertos

### J12 `1349123`

Ultimo estado: `RUNNING` en la V100 `r2n6`. El proceso estaba instalando PyTorch 2.5.1 y las dependencias CUDA en `UIE_TRAIN_VENV`:

```text
/home/y064/SCRATCH/y064644/paper3/venvs/uie-train
```

El resultado definitivo no se conoce. Este cierre no escribe PASS, no escribe FAIL y no rellena `preflight_report.json`. Los artefactos que el script versionado solo debe dejar tras un exit 0 son:

```text
/home/y064/SCRATCH/y064644/paper3/runs/p3/uie/preflight/environment_fingerprint.json
/home/y064/SCRATCH/y064644/paper3/runs/p3/uie/preflight/preflight_report.json
```

`preflight_report.json` tiene que traer `status=PASS` y un `fingerprint_sha256` para que el contrato de J13 pueda siquiera compararse. J13, 200 epocas, permanece `NO AUTORIZADO` aunque ese informe aparezca mas tarde. Autorizarlo es una decision posterior, fuera de este cierre.

`scripts/cesvima/bootstrap_uie_train_venv.sh` sale con `PASS reuse` en cuanto existe `bin/python` y no reinstala. Si `1349123` creo el venv y se interrumpio durante `pip`, un reenvio posterior puede saltarse la instalacion incompleta. Comprobar `torch==2.5.1+cu121` antes de aceptar ese reuse. Borrar `uie-train` no forma parte de este checkpoint.

El codigo upstream de UIE sigue sin parchear. El entorno historico del paper es Python 3.8, PyTorch 1.8.0, Transformers 4.6.1 y CUDA 10.2/11.1. El preflight moderno es una comprobacion de compatibilidad, no una declaracion de que el trainer upstream ya es compatible.

### GoLLIE `1349032`

Ultimo estado: `RUNNING`, aproximadamente 1 h 34 min. Se estaba compilando `flash-attn==2.8.3.post2` desde fuente porque el wheel precompilado era incompatible con la GLIBC del nodo. CUDA 12.1.1 y GCC 12.3 tenian la comprobacion del compilador aprobada. No hay evidencia final de `GOLLIE_SOURCE_BUILD_PASS` ni de `GOLLIE_WHEEL_ABI_PASS`.

Esas dos cadenas son criterios de la sesion. No las emite `scripts/cesvima/j15_gollie_preflight.py`. El PASS versionado, cuando exista, es otro artefacto:

```text
/home/y064/SCRATCH/y064644/paper3/runs/p3/gollie/preflight_report.json
```

con `status=PASS` y `weights_downloaded=false`. Hasta entonces no se instala ningun wheel y no se liberan J15, J16 ni J17. J16 no debe descargar GoLLIE-7B sin ese PASS.

El bootstrap versionado instala `flash-attn` sin pin y con `--no-build-isolation`, y tambien reutiliza el venv si `bin/python` ya existe:

```text
/home/y064/SCRATCH/y064644/paper3/venvs/gollie-l40s
```

El pin `2.8.3.post2` observado en el job no esta en el checkout local. No se incorpora en este cierre: cambiar el bootstrap seria un cambio de contrato sin un resultado de ABI.

Los fallos anteriores de J15 se conservan como evidencia. Sus JOBID no estan en el workspace local. No se borran logs para dejar sitio al intento `1349032`.

Revision de pesos que J16 usaria solo despues del PASS de J15:

```text
HiTZ/GoLLIE-7B
d3e41fef45f6a7d438c46ba7d9fce5d0d486c7a9
```

## Runbook al reconectar

Ejecutar solo desde el nodo de login ya usado. No relanzar jobs en el mismo paso en que se leen. No borrar venvs, logs ni el directorio de recuperacion.

```bash
sacct -j 1349043,1348833,1349096,1349121,1348832,1349123,1349032 \
  --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList,End
squeue -j 1349123,1349032 -o '%.18i %.28j %.10T %.12M %.6D %R'
```

Si un job sigue `RUNNING` o `PENDING`, el cierre de esta jornada se mantiene: no hay PASS nuevo.

Si `1349123` ya termino:

```bash
SCRATCH=/home/y064/SCRATCH/y064644/paper3
echo "--- shot pointer ---"
cat "$SCRATCH/runs/p3/uie/shot_dir.txt"
echo "--- preflight ---"
python3 -c 'import json,pathlib; p=pathlib.Path("'"$SCRATCH"'/runs/p3/uie/preflight/preflight_report.json"); print(p.read_text() if p.is_file() else "ABSENT")'
"$SCRATCH/venvs/uie-train/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Criterio de lectura de J12:

- Exit 0, `status=PASS` y fingerprint presente: preflight tecnico observado. Seguir sin enviar J13.
- Exit distinto de 0, informe ausente o `status` distinto de `PASS`: J12 no ha pasado. El venv puede haber quedado a medias.
- `shot_dir.txt` tiene que terminar en `seed1/10shot` y caer dentro de `uie-frozen-recovery-20261026`. Otra ruta vuelve a mezclar la reconstruccion fallida con los bytes recuperados.
- Confirmar `conll04-test-0` en `runs/p3/uie/smoke_example.json` sin copiar el texto fuera del cluster.

Si `1349032` ya termino:

```bash
SCRATCH=/home/y064/SCRATCH/y064644/paper3
python3 -c 'import json,pathlib; p=pathlib.Path("'"$SCRATCH"'/runs/p3/gollie/preflight_report.json"); print(p.read_text() if p.is_file() else "ABSENT")'
# Los marcadores de sesion no estan en el codigo versionado.
# Buscarlos en el log real del job, no darlos por escritos.
grep -E 'GOLLIE_SOURCE_BUILD_PASS|GOLLIE_WHEEL_ABI_PASS|flash-attn' \
  "$SCRATCH/runs/p3/slurm/"*1349032* 2>/dev/null || true
```

El `grep` solo cubre el directorio de logs del submitter versionado. Un `sbatch` manual deja `slurm-1349032.out` en el directorio desde el que se envio, y ese directorio no consta en el checkout local. J21 tiene la misma incertidumbre: su sbatch no fija `#SBATCH --output`.

Criterio de lectura de GoLLIE:

- Sin las dos cadenas de sesion y sin `preflight_report.json` en PASS: J15 sigue cerrado. No instalar un wheel y no enviar J16 ni J17.
- PASS del informe versionado con `weights_downloaded=false`, mas ABI e importacion comprobadas: recien entonces se puede plantear J16. Este runbook no lo envia.
- Conservar los logs de los J15 anteriores junto al de `1349032`.

Relectura de J21, sin relanzar Gold:

```bash
python3 -c 'import json,pathlib; p=pathlib.Path("/home/y064/SCRATCH/y064644/paper3/runs/p3/qwen3/invented-smoke/1349043"); print(sorted(x.name for x in p.iterdir()) if p.is_dir() else "ABSENT")'
```

Hace falta ver `technical_pass=true` y `gold_inference_authorized=false` en el informe del propio job. Si el directorio no esta en esa ruta, el job se envio con otro `--output` o con el `slurm-1349043.out` por defecto. No se lanza `--execute-c0-c1`: el script lo rechaza.

## Riesgos abiertos

1. Entorno UIE parcialmente instalado. `1349123` estaba en `pip` de PyTorch 2.5.1. El bootstrap reutiliza un `bin/python` existente y puede dejar el venv a medias sin decirlo en un reenvio.
2. Incompatibilidad historica del codigo UIE. El trainer upstream pertenece a Python 3.8 / PyTorch 1.8 / Transformers 4.6. El preflight moderno aun no tiene resultado. Un import o un batch roto se registra como FAIL de J12 y no se parchea el upstream para forzarlo.
3. FlashAttention y GLIBC. El wheel precompilado no sirvio en el nodo. La compilacion `flash-attn==2.8.3.post2` no tiene cierre. El bootstrap del repositorio ni siquiera pinea esa version. J16 y J17 siguen cerrados.
4. Solapamientos de CoNLL04. Los cinco solapes de texto y los duplicados internos siguen vigentes. Un entrenamiento futuro que los oculte dejararia de ser la reproduccion congelada.
5. Bloqueo Gold de Qwen3. `technical_pass=true` en documentos inventados no autoriza C0/C1 sobre Hohfeld. El SHA-256 del indice no esta aprobado y la anomalia de `total_size` sigue pendiente de aprobacion.
6. Dos cadenas UIE conviven. La recuperacion alimenta `shot_dir.txt`. La cola antigua sigue esperando a `1348833`. Reanudar la cola antigua reactivaria la reconstruccion fallida.
7. `public_redistribution=UNRESOLVED`. Los derivados CoNLL04, los pesos, los logs completos y `recovery_registry.json` se quedan en SCRATCH y en `_legacy`. No entran en git.

## Documentos que este cierre no reescribe

- `artifacts/uie_conll04_10shot_seed1_manifest.json` y `artifacts/uie_conll04_10shot_seed1_validation.json`. Siguen siendo el freeze.
- `docs/UIE_CONLL04_REPRODUCTION.md`, seccion 12. Conserva el estado P0 de septiembre (`CESVIMA wrapper: NOT CREATED`, `ready to submit job: NO`). Esta superado por la capa de jobs de octubre, pero es el registro historico de aquel freeze. El estado operativo del 26 de octubre es este fichero mas `docs/CESVIMA_TRAIN_FIRST.md`.
- `scripts/cesvima/bootstrap_gollie_venv.sh` y `bootstrap_uie_train_venv.sh`. El pin de FlashAttention y la reparacion de un venv a medias quedan para cuando se vea el exit real.
- Contratos C0/C1, shell v2, splits, etiquetas y esquemas.

## Reanudacion minima

1. Leer `sacct` de los siete JOBID de la tabla. No inferir el estado de `1349123` ni de `1349032` antes de eso.
2. Si J12 termino, archivar el informe de preflight y la version exacta de torch. Mantener J13 sin enviar.
3. Si GoLLIE termino, archivar el log de compilacion, las dos cadenas de sesion si aparecen, y el informe versionado si existe. Mantener J16 y J17 sin enviar.
4. Dejar `1348833` como FAILED en el registro. No borrar su log.
5. Gold Qwen3 solo se puede plantear en otra decision, con el SHA del indice aun sin aprobar y con el pase tecnico limitado a los cuatro textos inventados.

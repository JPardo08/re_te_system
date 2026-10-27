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

GoLLIE, transiciones. `BUILD_PASS` es la compilacion del wheel y no acredita `gollie-l40s`. La identidad binaria compara version, SHA-256 del wheel y SHA-256 de la extension contenida en el. El hash `1fed4874…` queda fuera de `MATCH` aunque alguien afirme que el wheel lo contiene. `ABI_PASS` exige, ademas de `MATCH`, que las dependencias dinamicas se resuelvan y que no falten simbolos GLIBC, observado en el nodo GPU. El kernel minimo solo cuenta sobre L40S y despues de esa ABI. `PREFLIGHT_PASS` exige las seis condiciones y el JOBID del informe. Hoy `current_artifacts_allow_preflight` es false. Un `PREFLIGHT_PASS` futuro no pone `GOLLIE_J16_DOWNLOAD_AUTHORIZED`. La recuperacion en un venv nuevo sigue deshabilitada y no modifica `gollie-l40s`.

`bootstrap_gollie_venv.sh` no contiene `pip` en codigo ejecutable. Si el venv no existe, sale 1. Si existe, evalua versiones y la identidad de la extension. Exit 0 significa `MATCH` de la extension con el wheel pineado. No significa `ABI_PASS` ni `PREFLIGHT_PASS`. Exit 2 cubre `BINARY_MISMATCH` y comparacion ausente, sin reinstalar y sin compilar. J15 no lanza el kernel si la identidad o la ABI no estan acreditadas. Escribe el informe en un directorio de JOBID y solo entonces, con `PREFLIGHT_PASS`, el puntero de ese JOBID. J16 ignora `gollie/preflight_report.json`, rechaza otro JOBID y, aun cuando `j16_rejection_reason` no encuentra un defecto tecnico, no descarga porque `GOLLIE_J16_DOWNLOAD_AUTHORIZED` es false. J17 sale si `GOLLIE_J17_SMOKE_AUTHORIZED` no vale `1`. `env.sh` no define esa variable.

`env.sh` y `j00_verify_infer_env.py` no forman parte de este cambio. En el historial local son identicos entre `a65fb07` y `67205d4`. Las modificaciones de esos dos ficheros en el checkout remoto son trabajo que hay que preservar y revisar para una posible integracion; no se descartan y no se sobrescriben.

## Sincronizacion, sin ejecutarla

Estado de partida conocido: CESVIMA en `a65fb07`, con `env.sh` y `j00_verify_infer_env.py` modificados y con scripts y logs historicos sin seguimiento. El local de esta recuperacion esta por delante de `67205d4` y todavia no esta commiteado.

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

## Commits propuestos, sin crearlos

1. `gollie: pin the FlashAttention wheel and keep the pre-correction binary unaccredited`
2. `uie: pin the origin checkpoint and keep every later phase disabled`
3. `docs: record the EXP.4.5.1 transition gates and the conservative sync`

El primero cubre `flash_attn_pin.py`, `gollie_environment.py`, el bootstrap, J15, J16, J17, `tests/test_exp45_gollie_recovery.py` y `tests/test_exp451_gollie_transitions.py`. El segundo cubre el pin de origen, la conversion deshabilitada, `uie_recovery_phases.py`, el fingerprint, J12, J13, `tests/test_exp45_uie_recovery.py` y `tests/test_exp451_uie_transitions.py`. El fichero mixto `tests/test_exp45_recovery.py` ya no existe: sus negativos quedaron partidos en esos dos ficheros. El tercero cubre este fichero y `docs/CESVIMA_TRAIN_FIRST.md`. No incluyen manifiestos cientificos ni el clon upstream.

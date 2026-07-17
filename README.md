# Experimento empírico: preservación de información en *cruces* de modelos de lenguaje

---

## 1. Introducción

En este trabajo se mide empíricamente el factor de degradación de información $\alpha$ entre modelos de lenguaje reales y se estudia su efecto en la dinámica de calidad del proyecto.

En el informe, $\alpha$ —que gobierna cuánta información se pierde de una generación a la siguiente— era un **parámetro libre** de la simulación. Aquí se mide entre cuatro modelos reales y, en lugar de limitarse al lazo recursivo de un modelo sobre sí mismo ($A \to A$), se analizan **todos los cruces** $A \to B$: qué ocurre cuando el texto que produce un modelo es "leído" por otro distinto.

La motivación del trabajo es cuantificar cuánta información sobrevive cuando la salida de un modelo alimenta a otro —el fenómeno detrás del *model collapse* y de la degradación de corpus cuando el contenido generado se recicla—. Se usa texto humano real (SQuAD) como referencia de máxima calidad ($I = 1$) y como estímulo inicial de las generaciones.

---

## 2. Definición del problema

Cuando un modelo $A$ genera un texto $x_A$, se busca determinar **qué fracción de su información sobrevive** al ser evaluada por otro modelo $B$. El problema tiene tres partes encadenadas:

- **Medir** la degradación de cada cruce $A \to B$ a partir de la cross-entropy que cada modelo asigna al texto del otro.
- **Inyectar** cada $\alpha_{A\to B}$ medido en la recurrencia de calidad del informe y obtener una calidad estacionaria $I^\star$ por cruce.
- **Comparar** los cruces y los modelos para responder la pregunta central del proyecto: **qué modelo preserva mejor el texto humano** y, en general, qué modelo conserva mejor la información ajena.

Se trata de un **experimento de medición** sobre modelos ya entrenados: $\alpha$ no se postula como parámetro, se estima directamente a partir de los textos que los modelos generan.

---

## 3. Formulación matemática

### Medida de información (Shannon)

Cuando el modelo $M$ ve un texto $x$, la cross-entropy que le asigna es

$$H_M(x) = -\frac{1}{N}\sum_{i=1}^{N}\log P_M(x_i \mid x_{<i}),$$

en nats/token. Es la cantidad shannoniana de información: si $M$ encuentra $x$ muy sorprendente, le cuesta más describirlo ⇒ perdió información sobre la distribución que generó $x$.

### Factor de degradación del cruce

$$\boxed{\;\alpha_{A\to B} = \frac{H_A(x_A)}{H_B(x_A)} \in (0,1]\;}$$

donde $x_A$ es texto **generado por $A$**. $H_A(x_A)$ es la auto-entropía de $A$ (el mínimo posible para su propio texto) y $H_B(x_A)$ es lo que le cuesta a $B$ ese mismo texto. El cociente mide qué fracción de la información de $A$ sobrevive al ser vista por $B$. Es **asimétrico** por construcción: $\alpha_{A\to B} \neq \alpha_{B\to A}$.

### Dinámica de calidad

Cada $\alpha_{A\to B}$ medido se inyecta en la recurrencia escalar del informe

$$I_{t+1} = h + (1-h)\big[(1-f)\,\alpha_{A\to B}\,I_t + f\big],$$

cuyo valor estacionario es

$$I^\star = \frac{h + (1-h)f}{1 - (1-h)(1-f)\,\alpha}.$$

Se usa $h = 0.05$ (inyección humana mínima) y $f = 0$ (sin curaduría, para aislar el efecto del cruce), con $T = 200$ iteraciones. El cruce con mayor $\alpha$ (y por tanto mayor $I^\star$) es el que mejor preserva la información.

Por último, la matriz de degradación medida se reutiliza como **matriz de mezcla** $W = \alpha$ (normalizada por filas) en el modelo vectorial acoplado del informe, cuya condición de estabilidad es $(1-h)(1-f)\,\rho(W) < 1$, con $\rho$ el radio espectral de $W$.

---

## 4. Objetivos

A partir de las matrices $\alpha$ e $I^\star$, el experimento responde cuatro preguntas:

1. **Quién preserva mejor el texto humano**: $\alpha_{\text{humano}\to j}$ por evaluador $j$ (la pregunta central del proyecto).
2. **Mejor evaluador**: qué modelo conserva mejor la información ajena, en promedio sobre todos los generadores.
3. **Mejor generador**: qué modelo produce texto que sobrevive mejor al ser evaluado por otros.
4. **Mejor y peor cruce individual**, y verificación de la estabilidad del modelo vectorial acoplado.

---

## 5. Datos y modelos

### Modelos evaluados

| Etiqueta | HuggingFace ID | Parámetros |
|---|---|---|
| `pythia70m` | `EleutherAI/pythia-70m` | 70 M |
| `distilgpt2` | `distilgpt2` | 82 M |
| `smollm2` | `HuggingFaceTB/SmolLM2-135M` | 135 M |
| `qwen0.5b` | `Qwen/Qwen2.5-0.5B` | 500 M |

### Datos: SQuAD

Se usa el split de **validación** de SQuAD (`rajpurkar/squad`, 10 570 ejemplos) de dos formas:

- **Prompts-semilla**: las primeras 8 palabras de cada `context` real arrancan la generación de los modelos (estímulo humano variado y reproducible), en vez de frases inventadas.
- **Referencia humana $I = 1$**: se guardan los `context` completos (recortados a 70 palabras) como corpus humano $x_{\text{hum}}$. Esto materializa el supuesto S1 del informe ($I_{\text{hum}} = 1$) y permite medir $\alpha_{\text{humano}\to j}$.

Se seleccionan 8 contextos (de más de 40 palabras), con 2 generaciones por prompt y 60 tokens nuevos por generación.

---

## 6. Resultados

> **Nota sobre las imágenes:** las figuras las genera el notebook en `salidas_cruces_llm/`. Copia los 5 `.png` a la carpeta `images/` (con los mismos nombres) para que se muestren aquí.

### Matriz de degradación $\alpha_{i\to j}$

Filas = generador $i$, columnas = evaluador $j$ (1.000 = sin pérdida):

| generador \ evaluador | pythia70m | distilgpt2 | smollm2 | qwen0.5b |
|---|---|---|---|---|
| **pythia70m**  | 1.000 | 0.837 | 0.918 | 0.835 |
| **distilgpt2** | 0.965 | 1.000 | 1.145 | 1.033 |
| **smollm2**    | 0.707 | 0.725 | 1.000 | 0.823 |
| **qwen0.5b**   | 0.596 | 0.651 | 0.798 | 1.000 |
| **humano**     | 0.619 | 0.700 | 0.802 | 1.000 |

Lectura de las cuatro preguntas:

- **Preservación del texto humano** (fila `humano`): `qwen0.5b` $\alpha = 1.000$ > `smollm2` $0.802$ > `distilgpt2` $0.700$ > `pythia70m` $0.619$. El modelo más grande es el que mejor conserva el texto humano.
- **Mejor evaluador** ($\alpha$ entrante medio): `qwen0.5b` $0.923$ ≈ `smollm2` $0.916$ > `distilgpt2` $0.728$ ≈ `pythia70m` $0.722$.
- **Mejor generador** ($\alpha$ saliente medio): `distilgpt2` $1.048$ > `pythia70m` $0.863$ > `smollm2` $0.752$ > `qwen0.5b` $0.682$. El texto de `distilgpt2` es el que mejor sobrevive a otros modelos.
- **Mejor / peor cruce**: mejor `distilgpt2 → smollm2` ($\alpha = 1.145$); peor `qwen0.5b → pythia70m` ($\alpha = 0.596$).

### Figuras

![Mapas de calor de α y de la calidad estacionaria I*](<img width="1462" height="607" alt="image" src="https://github.com/user-attachments/assets/972cfa35-9819-49ed-8b74-756aeffbcc3e" />
)

*Matriz de degradación $\alpha_{i\to j}$ (1 = sin pérdida) y calidad estacionaria $I^\star_{i\to j}$. Filas: generadores (4 modelos + humano); columnas: evaluadores.*

![Trayectorias de todos los cruces](<img width="1048" height="554" alt="image" src="https://github.com/user-attachments/assets/f142367c-6d5b-4cdf-b09a-59dc06d15526" />
)

*Evolución $I_t$ de cada cruce $i\to j$; los cruces `humano→` van en línea gruesa. Las curvas que se estabilizan más arriba preservan mejor la información.*

![Trayectorias agrupadas por generador](<img width="1587" height="785" alt="image" src="https://github.com/user-attachments/assets/428d5e53-41d6-4ed8-be2a-08bc7f8f3ccb" />
)

*Una subgráfica por generador: cómo le va a su texto según qué modelo lo procese.*

![Ranking de preservación](<img width="1318" height="457" alt="image" src="https://github.com/user-attachments/assets/ec65b1fe-8c0a-4964-90ec-5ee82a90262e" />
)

*$\alpha$ entrante medio (mejor evaluador / preserva mejor lo ajeno) y $\alpha$ saliente medio (su texto sobrevive mejor).*

![Modelo vectorial acoplado](<img width="1203" height="584" alt="image" src="https://github.com/user-attachments/assets/5f2d5296-8026-4efc-9b16-9dcc417dd9a4" />
)

*Modelo vectorial acoplado con la matriz empírica $W = \alpha$ (normalizada por filas): trayectoria $I_t^{(k)}$ de cada modelo.*

---

## 7. Conclusiones

- El modelo más grande (`qwen0.5b`, 500 M) es el **mejor evaluador** y el que **mejor preserva el texto humano** ($\alpha_{\text{humano}\to\text{qwen}} = 1.000$): a mayor capacidad, menor sorpresa ante texto ajeno.
- Ser buen evaluador **no** implica ser buen generador: `qwen0.5b` es el peor generador ($\alpha$ saliente $0.682$), mientras que `distilgpt2` —mucho más pequeño— produce el texto que mejor sobrevive a otros modelos. La preservación es una propiedad **asimétrica**.
- La asimetría $\alpha_{A\to B} \neq \alpha_{B\to A}$ se confirma en toda la matriz: leer y ser leído no son intercambiables.
- Algunos cruces dan $\alpha > 1$ (p. ej. `distilgpt2 → smollm2` $= 1.145$): el evaluador encuentra el texto ajeno **menos** sorprendente que el propio generador su propio texto. Es decir, la auto-entropía no siempre es el mínimo global, lo que rompe el supuesto $\alpha \in (0,1]$ de la recurrencia y produce valores de $I^\star$ fuera de rango en esos cruces —un límite a tener en cuenta al mapear la medida empírica al modelo teórico.
- El modelo vectorial acoplado con la matriz empírica $W$ resulta estable ($(1-h)(1-f)\,\rho(W) < 1$), de modo que la calidad no colapsa a cero bajo la mezcla de cruces medida.

---

## 8. Referencias

- Informe del proyecto — base teórica de la recurrencia $I_{t+1}$, el supuesto $I_{\text{hum}} = 1$ y el modelo vectorial acoplado. *(Añadir cita/enlace del informe.)*
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP. https://arxiv.org/abs/1606.05250 · Dataset: https://huggingface.co/datasets/rajpurkar/squad
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal.
- Modelos (HuggingFace): [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) · [distilgpt2](https://huggingface.co/distilgpt2) · [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) · [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Hugging Face *Transformers* — https://github.com/huggingface/transformers

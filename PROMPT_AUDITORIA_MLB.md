# ROL Y MISIÓN

Actúa como un **Senior Staff Engineer + Senior ML Scientist (top 1%)** especializado en sistemas de optimización combinatoria, modelado estadístico de baseball (sabermetría) y aplicaciones de datos en producción. Tienes décadas de experiencia revisando código de data scientists junior y convirtiendo prototipos en sistemas robustos.

En el directorio actual hay un proyecto de un **data scientist junior** cuyo objetivo es **encontrar la mejor alineación posible de bateadores (batting order) para un partido de la MLB**. Tu trabajo NO es reescribir el proyecto todavía. Tu trabajo es **entenderlo a la perfección de principio a fin y producir una auditoría exhaustiva** con una lista priorizada de mejoras, debilidades, errores y oportunidades.

Trabaja con rigor: no asumas nada que no hayas verificado leyendo el código. Si algo no lo puedes confirmar, márcalo explícitamente como "no verificado" en vez de inventarlo.

---

# FASE 0 — DESCUBRIMIENTO Y SKILLS (obligatorio antes de auditar)

1. **Antes de leer cualquier archivo de datos o escribir código, revisa las skills disponibles** en el entorno (p. ej. `/mnt/skills/...` y skills de usuario). Para CUALQUIER acción que vaya a producir o tocar un archivo (`.docx`, `.pdf`, `.xlsx`, `.pptx`, `.csv`) o ejecutar análisis tabular, **abre y lee primero el `SKILL.md` correspondiente** y síguelo. En concreto:
   - Entregable de auditoría en Word → skill **docx**.
   - Si exportas la lista de mejoras o un modelo a hoja de cálculo → skill **xlsx**.
   - Si lees/escribes PDFs → skill **pdf** / **pdf-reading**.
   - Si generas una presentación de resultados → skill **pptx**.
   - Para entender ficheros subidos cuyo contenido no esté en contexto → skill **file-reading**.
   - Para cualquier cambio de UI/frontend → skill **frontend-design**.
   - Para detalles de productos Anthropic → skill **product-self-knowledge**.
   Lee todas las que apliquen (pueden aplicar varias a la vez). No te saltes este paso aunque creas que dominas el formato.

2. **Mapea el repositorio completo** antes de opinar: árbol de directorios, ficheros de entrada (datos), notebooks, scripts, módulos, `requirements.txt`/`pyproject.toml`/`environment.yml`, configuración, tests, `.git` (historial y ramas si existe), README, y la app de Streamlit. Identifica el punto de entrada y traza el flujo de datos de extremo a extremo.

---

# FASE 1 — COMPRENSIÓN PROFUNDA DE EXTREMO A EXTREMO

Reconstruye y documenta, en tus propias palabras, **toda la tubería**:

1. **Ingesta de datos**: ¿De dónde salen los datos (CSV local, API tipo pybaseball/Statcast/Retrosheet/Lahman, scraping)? ¿Qué temporada(s), qué nivel (jugador, equipo, situación)? ¿Cómo se cargan, cachean y validan?
2. **Limpieza y features**: ¿Qué métricas usa (AVG, OBP, SLG, OPS, wOBA, wRC+, ISO, BABIP, K%, BB%, splits vs LHP/RHP, situacional con corredores en base)? ¿Calcula bien cada fórmula? ¿Maneja muestras pequeñas, NaNs, jugadores sin suficientes apariciones, regresión a la media?
3. **Modelo / motor de optimización**: ¿Cómo decide la mejor alineación?
   - ¿Es un **ranking heurístico** (ordenar por OPS/OBP), una **búsqueda combinatoria** (las 9! = 362.880 permutaciones, o subconjuntos), un **modelo de simulación** (Markov de estados base-outs / Monte Carlo de la entrada), **programación lineal/entera**, o **algoritmo genético/metaheurística**?
   - ¿La **función objetivo** es la correcta? (maximizar carreras esperadas por juego/entrada, no solo "suma de OPS"). Evalúa si la métrica que optimiza realmente corresponde a ganar partidos.
   - ¿Modela el orden real del baseball (el bateador 1 batea más veces que el 9; importancia del 2-3-4; interacción entre bateadores consecutivos; OBP arriba, slugging en medio)?
4. **Evaluación**: ¿Cómo valida que una alineación es mejor que otra? ¿Backtesting, validación cruzada, comparación contra la alineación real del equipo, intervalos de confianza?
5. **Salida y presentación**: ¿Qué entrega al usuario y cómo lo visualiza en Streamlit?

Entrega un **diagrama de flujo** del pipeline (texto/mermaid) y una explicación clara de la lógica de optimización tal y como está implementada hoy.

---

# FASE 2 — AUDITORÍA EXHAUSTIVA

Revisa **todas y cada una de las cosas que hace el proyecto**. No te limites a lo obvio. Cubre, como mínimo, estas dimensiones, y en cada hallazgo indica: **[archivo:línea] · severidad (Crítica/Alta/Media/Baja) · impacto · esfuerzo de arreglo · recomendación concreta (con snippet si aplica)**.

### A. Correctitud del dominio (sabermetría)
- Fórmulas estadísticas correctas y actualizadas (p. ej. pesos de wOBA por año, no constantes obsoletas).
- ¿Optimiza carreras esperadas o un proxy débil? ¿Ignora el efecto del orden de turnos (plate appearance leverage)?
- Tratamiento de splits (vs zurdos/diestros), park factors, lineup protection, corredores en base, situacional.
- Tamaño muestral y regresión a la media para jugadores con pocas apariciones.
- ¿Usa datos de la temporada/oponente/pitcher correctos para el partido objetivo?

### B. Correctitud del modelo / optimización
- ¿La función objetivo refleja el objetivo real? ¿Hay fuga de datos (data leakage)?
- Si es fuerza bruta sobre 9!: ¿es viable en tiempo? Si es heurística: ¿qué optimalidad sacrifica?
- ¿Hay un **modelo de simulación de entradas** (cadena de Markov de 24/25 estados base-outs o Monte Carlo)? Si no, propón añadirlo: es el estándar de oro para evaluar batting orders.
- Determinismo/reproducibilidad: semillas aleatorias fijadas, resultados estables.
- Validación estadística: ¿las diferencias entre alineaciones son significativas o ruido?

### C. Calidad de software (ingeniería)
- Estructura del proyecto, separación de capas (datos / lógica / UI), funciones puras vs efectos laterales.
- Legibilidad, naming, duplicación, números mágicos, funciones gigantes, notebooks-script sin modularizar.
- Manejo de errores y casos límite (jugador inexistente, datos faltantes, equipo vacío, API caída).
- Tipado (type hints), docstrings, configuración externalizada vs hardcodeada.
- **Rendimiento**: vectorización con pandas/numpy en vez de bucles, complejidad algorítmica, cuellos de botella, uso de memoria, caché de cómputos caros.
- **Tests**: ¿existen? Cobertura de la lógica crítica (fórmulas, optimizador). Propón tests unitarios mínimos.
- **Reproducibilidad/entorno**: dependencias pinneadas, versión de Python, instrucciones de ejecución, datos versionados o descargables.
- **Seguridad/robustez**: claves de API expuestas, validación de entradas del usuario, rutas hardcodeadas.

### D. Datos
- Fuente fiable y actual, frescura de los datos, caché con expiración, manejo de rate limits si hay API.
- Validación de esquema, detección de outliers, jugadores duplicados o mal mapeados.

### E. UX / Producto
- ¿El usuario entiende *por qué* esa es la mejor alineación? ¿Hay explicabilidad (contribución de cada jugador, carreras esperadas, comparación contra el orden actual)?

Termina la Fase 2 con una **tabla resumen priorizada** (matriz impacto × esfuerzo) y un **Top 10 de acciones de mayor ROI**.

---

# FASE 3 — MEJORAS DE LA INTERFAZ STREAMLIT

Audita y propón mejoras concretas para la app de Streamlit (sigue la skill **frontend-design** para las decisiones visuales). Cubre:

- **Arquitectura Streamlit**: uso de `st.cache_data`/`st.cache_resource` para no recomputar datos/optimización en cada rerun; `st.session_state` para estado; evitar trabajo pesado en el hilo de render; `st.form` para inputs agrupados; fragmentación/lazy-loading de secciones costosas.
- **Visualización del lineup**: diagrama del orden de bateo, campo de baseball con posiciones, comparativa lado a lado (alineación propuesta vs actual) con delta de carreras esperadas, gráfico de contribución por jugador, intervalos de confianza.
- **Interactividad**: selección de equipo/temporada/pitcher rival, filtros vs LHP/RHP, drag-and-drop o reordenado manual para que el usuario pruebe órdenes y vea el impacto en vivo, botón de "optimizar".
- **Explicabilidad y confianza**: panel que justifique la recomendación, métricas clave por jugador, tooltips con definiciones sabermétricas.
- **Diseño y rendimiento percibido**: layout con columnas/tabs/containers, jerarquía visual, spinners/progress para cómputos largos, paleta y tipografía intencionadas (no defaults), responsividad, estados de carga y de error claros, accesibilidad (contraste, etiquetas).
- **Exportación**: descargar la alineación y el informe (CSV/PDF/imagen).

Da ejemplos de código Streamlit para las 3–5 mejoras de mayor impacto.

---

# FASE 4 — ROADMAP Y ENTREGABLE

1. Propón un **roadmap por fases**: (a) *Quick wins* (correcciones y mejoras de bajo esfuerzo y alto impacto), (b) *Mejoras estructurales* (refactor, tests, modularización), (c) *Saltos de capacidad* (motor de simulación Markov/Monte Carlo, splits y park factors, explicabilidad, validación estadística), (d) *Visión* (qué haría a este proyecto verdaderamente state-of-the-art).
2. Para cada ítem importante, incluye un **plan de implementación accionable** (qué tocar, cómo, criterio de "hecho").

**Formato del entregable:**
- Primero, un **resumen ejecutivo** en el chat (hallazgos clave, Top 10 acciones, veredicto general del estado del proyecto).
- Después, genera un **informe completo en un documento Word (.docx)** usando la skill **docx**, con: portada, índice, diagrama del pipeline, auditoría por dimensión con la tabla de severidades, sección de Streamlit, roadmap, y apéndice con snippets de los arreglos prioritarios. Guarda el archivo en el directorio de salida y preséntalo al final.
- (Opcional, si aporta valor) exporta la **lista priorizada de mejoras como `.xlsx`** (skill xlsx) para seguimiento tipo backlog.

---

# REGLAS DE TRABAJO

- **Verifica leyendo el código real**; cita siempre `archivo:línea`. Nada de afirmaciones genéricas sin evidencia.
- Sé **directo y crítico pero constructivo**: el objetivo es elevar el proyecto, no lucirte. Trata al autor con respeto.
- Prioriza por **impacto en el objetivo real** (ganar partidos / maximizar carreras esperadas), no por preferencias estéticas.
- No reescribas todavía el proyecto entero; entrega la **auditoría + plan**. Si propones código, que sean snippets ilustrativos de los arreglos clave.
- Distingue siempre **error objetivo** vs **mejora opcional** vs **decisión de diseño defendible**.
- Si encuentras un problema de correctitud sabermétrica o de fuga de datos, **márcalo como Crítico**: invalida los resultados aunque el código "funcione".

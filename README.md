## 📋 Descripción
Sistema unificado que analiza imágenes por características de color, muestra gráficos de análisis y permite al usuario elegir el número óptimo de clusters para organizar las imágenes automáticamente.

## 🚀 Uso Básico

### Comando Principal
```bash
python complete_clustering.py <ruta_carpeta>
```

### Ejemplos

#### **Análisis Completo**
```bash
# Análisis completo de todas las imágenes
python complete_clustering.py Raw_data

# Limitar a 1000 imágenes por carpeta (más rápido)
python complete_clustering.py Raw_data --max-images 1000

# Usar carpeta personalizada
python complete_clustering.py "C:/MisImagenes"

# Mover archivos en lugar de copiarlos
python complete_clustering.py Raw_data --no-copy
```

#### **🎯 Análisis Únicamente (Sin Organización)**
```bash
# Análisis completo pero sin organizar carpetas
python complete_clustering.py Raw_data --only-analysis

# Análisis rápido con muestra pequeña
python complete_clustering.py Raw_data --only-analysis --max-images 200

# Análisis en carpeta personalizada
python complete_clustering.py "C:/MisImagenes" --only-analysis

# Análisis con más puntos en t-SNE para mejor calidad visual
python complete_clustering.py Raw_data --only-analysis --tsne-samples 10000
```

## 📊 Proceso Completo

### 1. **Exploración de Datos**
- Descubre automáticamente la estructura de carpetas
- Cuenta imágenes disponibles
- Estima tiempo de procesamiento

### 2. **Carga y Extracción de Características**
- Carga imágenes automáticamente (BMP, PNG, JPG, JPEG)
- Maneja conversión RGBA → RGB con fondo blanco
- Redimensiona a 64x64 píxeles para consistencia
- **Extrae 23 características** avanzadas de color por imagen:
  - **RGB**: Media, desviación estándar, percentiles (Q1, Q3), mediana por canal
  - **Color dominante**: Promedio RGB de toda la imagen  
  - **Brillo y contraste**: Luminosidad y variabilidad
  - **HSV**: Matiz, saturación y valor promedio

### 3. **Normalización y Preprocesamiento**
- 🔧 **Normalización StandardScaler**: Convierte características a media=0, std=1
- 📊 **Información detallada**: Muestra rangos antes y después de normalización
- 🎯 **Consistencia**: Todas las características en la misma escala para clustering efectivo
- ⚡ **PCA optimizado**: Reducción a 10 componentes con random_state para reproducibilidad

### 4. **Análisis de Clusters Óptimos**
- Evalúa diferentes números de clusters (k=2 a k=15)
- Calcula métricas: Silhouette Score e Inercia sobre datos normalizados
- Genera gráfico de análisis: **`cluster_analysis.png`** con **dos visualizaciones**:
  - 📈 **Método del Codo (Elbow method)**: Muestra reducción de inercia
  - 📊 **Análisis Silhouette**: Identifica el k óptimo (línea roja)

### 5. **Selección Inteligente del Usuario**
- Sistema sugiere número óptimo basado en mejor Silhouette Score
- Usuario revisa gráficos generados para tomar decisión informada
- Permite elegir número final de clusters (2-20)
- Acepta sugerencia automática presionando Enter

### 6. **Clustering Final y Visualización t-SNE**
- Realiza clustering con el número seleccionado por el usuario
- Genera estadísticas de distribución por cluster
- **Crea visualización t-SNE FINAL**: **`final_tsne_clusters.png`**
  - Muestra distribución espacial real con colores por cluster
  - Incluye conteo de imágenes por cluster
  - Usa muestra de hasta 5000 puntos por defecto (configurable)

### 7. **Organización Automática** *(Solo modo completo)*
- Crea carpetas `cluster_0`, `cluster_1`, etc.
- Copia (o mueve) imágenes a sus respectivos clusters
- Genera reporte detallado
- Crea muestras visuales de cada cluster

## 🎯 **MODO ANÁLISIS ÚNICAMENTE**

### ¿Cuándo usar `--only-analysis`?
- ✅ **Solo necesitas** visualizaciones (no organizar miles de archivos)
- ✅ **Análisis exploratorio** rápido para decidir parámetros
- ✅ **Validar configuraciones** antes del procesamiento completo
- ✅ **Presentaciones** - Solo requieres los gráficos

### Proceso Completo pero Sin Organización:
1. **Carga imágenes** → 2. **Normalización** → 3. **PCA** → 4. **Análisis clusters** → 5. **Selección** → 6. **Clustering** → 7. **t-SNE**

### Salida Generada:
- 📊 **`cluster_analysis.png`** - Gráficos de análisis (Codo + Silhouette)  
- 🎪 **`final_tsne_clusters.png`** - Visualización t-SNE con k clusters especificado
- 📊 **Estadísticas en consola** - Distribución de clusters y métricas de calidad

## 📁 Archivos Generados

```
Clustered_Images_YYYYMMDD_HHMMSS/
├── cluster_0/                    # Imágenes del cluster 0
├── cluster_1/                    # Imágenes del cluster 1
├── cluster_N/                    # Imágenes del cluster N
├── cluster_0_sample.png          # Vista previa cluster 0
├── cluster_1_sample.png          # Vista previa cluster 1
├── cluster_N_sample.png          # Vista previa cluster N
└── clustering_report.txt         # Reporte detallado

# Archivos de análisis (en carpeta principal)
cluster_analysis.png              # Análisis de número óptimo de clusters
final_tsne_clusters.png           # Visualización t-SNE con clusters finales
```

## 🎛️ Opciones Disponibles

| Opción | Descripción | Ejemplo |
|--------|-------------|---------|
| `data_folder` | **Requerido**. Ruta a la carpeta con imágenes | `Raw_data` |
| `--max-images N` | Limita imágenes por carpeta (opcional) | `--max-images 500` |
| `--no-copy` | Mueve archivos en lugar de copiarlos | `--no-copy` |
| `--only-analysis` | **🎯 NUEVO**: Solo análisis (sin organizar carpetas) | `--only-analysis` |
| `--tsne-samples N` | **🎯 NUEVO**: Máx. puntos para t-SNE (default: 5000) | `--tsne-samples 10000` |
| `--help` | Muestra ayuda completa | `--help` |

## � Normalización de Datos

### ¿Por qué es Crucial la Normalización?
- **Diferentes escalas**: Las características RGB (0-255) vs HSV (0-1) tienen rangos muy diferentes
- **Dominancia por escala**: Sin normalización, las características con valores grandes dominan el clustering
- **Clustering efectivo**: K-means requiere características en la misma escala para funcionar correctamente
- **Consistencia**: StandardScaler garantiza media=0 y desviación estándar=1 para todas las características

### Información Mostrada:
```
🔧 Normalizando características de color...
   ✅ Características normalizadas (media=0, std=1)
   📊 Forma original: (240, 23)
   📊 Rango pre-normalización: [0.00, 255.00]
   📊 Rango post-normalización: [-3.86, 2.94]
```

## �📊 Características Extraídas

El sistema extrae **23 características** por imagen:

### RGB Básicas (15 características)
- **Canales R, G, B**: Media, desviación estándar, percentiles 25/75, mediana

### Color y Brillo (4 características)  
- **Color dominante**: Componentes R, G, B promedio
- **Brillo general**: Luminosidad promedio de la imagen

### Textura (1 característica)
- **Contraste**: Variabilidad en la imagen

### HSV (3 características)
- **Hue (Matiz)**: Tono promedio
- **Saturación**: Intensidad de color promedio  
- **Valor**: Brillo en espacio HSV

## 🎯 Interpretación de Resultados

### Silhouette Score
- **0.7 - 1.0**: Excelente separación
- **0.5 - 0.7**: Buena separación  
- **0.25 - 0.5**: Separación aceptable
- **< 0.25**: Separación pobre

### Distribución de Clusters
- **Clusters grandes**: Patrones de color comunes
- **Clusters pequeños**: Patrones únicos o atípicos

## 🔧 Troubleshooting

### Errores Comunes

1. **"La carpeta no existe"**
   ```bash
   # Verifica la ruta
   ls Raw_data  # Linux/Mac
   dir Raw_data  # Windows
   ```

2. **"No se encontraron imágenes"**
   - Verifica que hay archivos .bmp, .png, .jpg, .jpeg
   - Revisa subcarpetas si las imágenes están organizadas ahí

3. **"Memoria insuficiente"**
   ```bash
   # Usa menos imágenes por carpeta
   python complete_clustering.py Raw_data --max-images 500
   ```

4. **"Error en PCA/Clustering"**
   - Puede ocurrir con muy pocas imágenes
   - Intenta con al menos 50-100 imágenes

## 💡 Consejos de Uso

### Para Análisis Rápido
```bash
python complete_clustering.py Raw_data --max-images 200
```

### Para Análisis Completo
```bash
python complete_clustering.py Raw_data
```

### Para Conservar Espacio
```bash
python complete_clustering.py Raw_data --no-copy
```

## 📈 Flujo de Trabajo Recomendado

### 🚀 Proceso Paso a Paso
1. **Análisis rápido** primero con muestra pequeña:
   ```bash
   python complete_clustering.py Raw_data --max-images 60
   ```
2. **Revisar gráficos iniciales** (`cluster_analysis.png`):
   - Observar método del codo y análisis Silhouette
   - Notar el k óptimo sugerido (línea roja)
3. **Seleccionar número de clusters**:
   - Usar sugerencia del sistema (Enter) o elegir manualmente
   - Considerar interpretabilidad vs. calidad técnica
4. **Revisar visualización t-SNE final** (`final_tsne_clusters.png`):
   - Verificar que los clusters se ven bien separados
   - Comprobar distribución de tamaños
5. **Análisis completo** si satisfecho con resultados:
   ```bash
   python complete_clustering.py Raw_data
   ```

### ⚡ Para Resultados Rápidos
- **Prueba inicial**: 60-200 imágenes por carpeta
- **Análisis completo**: Sin límite de imágenes  
- **Solo visualización**: `--only-analysis --k N` para generar únicamente t-SNE
- **Conservar espacio**: Usar `--no-copy` para mover archivos

### 🎯 Modo Análisis Únicamente  
Cuando quieres análisis completo pero sin organizar archivos:
```bash
python complete_clustering.py Raw_data --only-analysis
```
**Ventajas:**
- ⚡ **Más rápido** - Sin organización de carpetas ni reportes pesados
- 📊 **Análisis completo** - Muestra gráficos de codo y Silhouette 
- 🤔 **Selección interactiva** - Eliges el k después de ver el análisis
- 🎪 **Visualizaciones** - Genera ambos gráficos (análisis + t-SNE)
- 🔬 **Exploración** - Perfecto para análisis exploratorio

### 🎨 Control de Calidad t-SNE
**Nuevo parámetro `--tsne-samples`** para controlar la calidad visual:

```bash
# Calidad estándar (5000 puntos - default)
python complete_clustering.py Raw_data --only-analysis

# Alta calidad para datasets grandes (10000 puntos)
python complete_clustering.py Raw_data --only-analysis --tsne-samples 10000

# Análisis rápido (1000 puntos)
python complete_clustering.py Raw_data --only-analysis --tsne-samples 1000
```

**¿Cómo elegir el número de puntos?**
- **1000-2000**: Análisis rápido, calidad básica
- **5000 (default)**: Balance perfecto calidad/velocidad 
- **10000+**: Máxima calidad para datasets grandes
- **Sin límite**: Usa todos los puntos (puede ser lento con >20k imágenes)

## 🎨 Interpretación de Gráficos

### 📊 Análisis Inicial (cluster_analysis.png)
Contiene **2 gráficos** para ayudarte a decidir el número óptimo de clusters:

1. **📈 Método del Codo (Elbow Method)**
   - Muestra cómo disminuye la inercia al aumentar clusters
   - Busca el "codo" donde la mejora se estabiliza
   - Línea roja vertical marca el k óptimo sugerido

2. **📊 Análisis Silhouette** 
   - Muestra calidad de separación para cada k
   - **Busca el pico más alto** = mejor separación
   - Línea roja vertical: k óptimo
   - Línea verde horizontal: mejor score alcanzado

### 🎪 Visualización Final (final_tsne_clusters.png)
Se genera **DESPUÉS** de seleccionar el número de clusters:

- **Distribución Espacial t-SNE**: Proyección 2D de las características de color
  - **Puntos del mismo color** = mismo cluster
  - **Clusters bien separados** = grupos distintos y claros  
  - **Solapamiento** = clusters con características similares
  - **Conteo por cluster** = número real de imágenes asignadas
  - **Muestra utilizada** = hasta 5000 puntos por defecto (configurable con `--tsne-samples`)

### 💡 Consejos de Interpretación
- **Clusters compactos y separados** = excelente agrupación
- **Clusters dispersos** = características muy variables
- **Muchos clusters pequeños** = datos muy diversos
- **Pocos clusters grandes** = patrones dominantes claros## 🎯 Casos de Uso

- **Screenshots de juegos**: Separar por tipo de pantalla/nivel
- **Imágenes médicas**: Agrupar por características visuales
- **Fotografías**: Organizar por paleta de colores
- **Documentos escaneados**: Separar por tipo de contenido
- **Arte digital**: Clasificar por estilo cromático

---

**¡El sistema está listo para usar! 🚀**

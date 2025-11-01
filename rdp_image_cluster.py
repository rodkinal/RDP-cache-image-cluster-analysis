#!/usr/bin/env python3
"""
🎯 CLUSTERING COMPLETO DE IMÁGENES
===================================

Script unificado que:
1. Lee tiles de imágenes desde una carpeta especificada
2. Realiza análisis exploratorio con gráficos
3. Permite al usuario elegir el número de clusters
4. Organiza las imágenes en carpetas separadas

Uso:
    python complete_clustering.py <ruta_carpeta_tiles>

Ejemplo:
    python complete_clustering.py Raw_data
    python complete_clustering.py "C:/Users/Usuario/MisDatos"

Autor: Rodkinal
Fecha: 2025
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from collections import Counter
import pandas as pd
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('default')
sns.set_palette("husl")

class CompleteImageClustering:
    """
    Sistema completo de clustering de imágenes con análisis y visualización
    """
    
    def __init__(self, data_folder, tsne_samples=5000):
        self.data_folder = data_folder
        self.tsne_samples = tsne_samples
        self.image_paths = []
        self.image_features = []
        self.image_names = []
        self.folder_labels = []
        self.cluster_labels = []
        self.pca_features = None
        self.scaler = None
        self.pca_model = None
        
        # Verificar que la carpeta existe
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"❌ La carpeta {data_folder} no existe")
    
    def discover_data_structure(self):
        """
        Descubre y analiza la estructura de datos
        """
        print(f"🔍 EXPLORANDO ESTRUCTURA DE DATOS")
        print("=" * 50)
        print(f"📁 Carpeta base: {self.data_folder}")
        
        # Buscar subcarpetas con imágenes
        folders_found = []
        total_images = 0
        
        for item in os.listdir(self.data_folder):
            item_path = os.path.join(self.data_folder, item)
            if os.path.isdir(item_path):
                # Buscar imágenes BMP en la subcarpeta
                image_files = [f for f in os.listdir(item_path) 
                              if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
                
                if len(image_files) > 0:
                    folders_found.append((item, len(image_files)))
                    total_images += len(image_files)
                    print(f"  📂 {item}: {len(image_files):,} imágenes")
        
        if not folders_found:
            # Si no hay subcarpetas, buscar directamente en la carpeta base
            image_files = [f for f in os.listdir(self.data_folder) 
                          if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
            if len(image_files) > 0:
                folders_found.append((".", len(image_files)))
                total_images = len(image_files)
                print(f"  📂 Carpeta raíz: {len(image_files):,} imágenes")
        
        if total_images == 0:
            raise ValueError("❌ No se encontraron imágenes en la carpeta especificada")
        
        print(f"\n📊 RESUMEN:")
        print(f"  🗂️ Carpetas con imágenes: {len(folders_found)}")
        print(f"  🖼️ Total de imágenes: {total_images:,}")
        
        # Estimación de tiempo
        time_estimate = total_images / 200  # Aproximadamente 200 imágenes por segundo
        print(f"  ⏱️ Tiempo estimado de procesamiento: {time_estimate/60:.1f} minutos")
        
        return folders_found, total_images
    
    def load_images(self, max_images_per_folder=None, sample_size=(64, 64)):
        """
        Carga todas las imágenes para análisis
        """
        print(f"\n🔄 CARGANDO IMÁGENES")
        print("=" * 30)
        
        folders_found, _ = self.discover_data_structure()
        
        for folder_name, img_count in folders_found:
            if folder_name == ".":
                folder_path = self.data_folder
            else:
                folder_path = os.path.join(self.data_folder, folder_name)
            
            # Obtener archivos de imagen
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
            
            if max_images_per_folder:
                image_files = image_files[:max_images_per_folder]
            
            print(f"  📂 {folder_name}: procesando {len(image_files)} imágenes...")
            
            for i, img_file in enumerate(image_files):
                img_path = os.path.join(folder_path, img_file)
                try:
                    # Cargar y procesar imagen
                    img = Image.open(img_path)
                    
                    # Manejar diferentes modos de imagen
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize(sample_size)
                    img_array = np.array(img)
                    
                    # Extraer características de color
                    features = self._extract_color_features(img_array)
                    
                    self.image_paths.append(img_path)
                    self.image_features.append(features)
                    self.image_names.append(img_file)
                    self.folder_labels.append(folder_name)
                    
                    # Mostrar progreso cada 1000 imágenes
                    if (i + 1) % 1000 == 0:
                        print(f"    📋 Procesadas {i + 1} imágenes...")
                    
                except Exception as e:
                    print(f"    ❌ Error cargando {img_file}: {e}")
                    continue
        
        self.image_features = np.array(self.image_features)
        print(f"\n✅ Cargadas {len(self.image_features):,} imágenes exitosamente")
        print(f"📊 Dimensiones de características: {self.image_features.shape}")
    
    def _extract_color_features(self, img_array):
        """
        Extrae características completas de color de una imagen
        """
        features = []
        
        # 1. Estadísticas básicas por canal RGB
        for channel in range(3):
            channel_data = img_array[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),           # Media
                np.std(channel_data),            # Desviación estándar
                np.percentile(channel_data, 25), # Q1
                np.percentile(channel_data, 75), # Q3
                np.median(channel_data),         # Mediana
            ])
        
        # 2. Color dominante (promedio de toda la imagen)
        dominant_color = np.mean(img_array.reshape(-1, 3), axis=0)
        features.extend(dominant_color)
        
        # 3. Brillo general
        brightness = np.mean(img_array)
        features.append(brightness)
        
        # 4. Contraste (desviación estándar del brillo)
        gray = np.mean(img_array, axis=2)
        contrast = np.std(gray)
        features.append(contrast)
        
        # 5. Conversión a HSV para características adicionales
        img_hsv = plt.cm.colors.rgb_to_hsv(img_array / 255.0)
        features.extend([
            np.mean(img_hsv[:, :, 0]),  # Hue promedio
            np.mean(img_hsv[:, :, 1]),  # Saturación promedio
            np.mean(img_hsv[:, :, 2]),  # Valor promedio
        ])
        
        return np.array(features)
    
    def prepare_data_and_analyze_clusters(self):
        """
        Prepara los datos y analiza directamente los clusters óptimos
        """
        print(f"\n📊 PREPARANDO DATOS Y ANALIZANDO CLUSTERS")
        print("=" * 50)
        
        if len(self.image_features) == 0:
            print("❌ No hay imágenes cargadas")
            return
        
        # 1. Normalizar características (StandardScaler)
        print("🔧 Normalizando características de color...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.image_features)
        print(f"   ✅ Características normalizadas (media=0, std=1)")
        print(f"   📊 Forma original: {self.image_features.shape}")
        print(f"   📊 Rango pre-normalización: [{self.image_features.min():.2f}, {self.image_features.max():.2f}]")
        print(f"   📊 Rango post-normalización: [{scaled_features.min():.2f}, {scaled_features.max():.2f}]")
        
        # 2. PCA para reducción de dimensionalidad
        print("🔧 Aplicando reducción de dimensionalidad (PCA)...")
        max_components = min(scaled_features.shape[0], scaled_features.shape[1])
        n_components = min(10, max_components - 1)
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)
        
        self.pca_features = pca_features
        self.scaler = scaler
        self.pca_model = pca
        
        print(f"   ✅ PCA aplicado con {n_components} componentes")
        print(f"   📊 Varianza explicada total: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"   📊 Dimensiones finales: {pca_features.shape}")
        
        # Análisis directo de clusters óptimos
        optimal_k = self._analyze_optimal_clusters(pca_features)
        
        return optimal_k
    
    def prepare_data_for_analysis_only(self):
        """
        Prepara los datos y muestra análisis de clusters para modo --only-analysis
        """
        print(f"\n📊 PREPARANDO DATOS Y ANALIZANDO CLUSTERS")
        print("=" * 50)
        
        if len(self.image_features) == 0:
            print("❌ No hay imágenes cargadas")
            return
        
        # 1. Normalizar características (StandardScaler)
        print("🔧 Normalizando características de color...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.image_features)
        print(f"   ✅ Características normalizadas (media=0, std=1)")
        print(f"   📊 Forma original: {self.image_features.shape}")
        print(f"   📊 Rango pre-normalización: [{self.image_features.min():.2f}, {self.image_features.max():.2f}]")
        print(f"   📊 Rango post-normalización: [{scaled_features.min():.2f}, {scaled_features.max():.2f}]")
        
        # 2. PCA para reducción de dimensionalidad
        print("🔧 Aplicando reducción de dimensionalidad (PCA)...")
        max_components = min(scaled_features.shape[0], scaled_features.shape[1])
        n_components = min(10, max_components - 1)
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(scaled_features)
        
        self.pca_features = pca_features
        self.scaler = scaler
        self.pca_model = pca
        
        print(f"   ✅ PCA aplicado con {n_components} componentes")
        print(f"   📊 Varianza explicada total: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"   📊 Dimensiones finales: {pca_features.shape}")
        
        # Análisis de clusters óptimos (igual que el modo completo)
        optimal_k = self._analyze_optimal_clusters(pca_features)
        
        return optimal_k

    
    def _analyze_optimal_clusters(self, pca_features, max_k=15):
        """
        Analiza el número óptimo de clusters
        """
        print(f"\n🎯 ANÁLISIS DE CLUSTERS ÓPTIMOS")
        print("-" * 40)
        
        k_range = range(2, min(max_k + 1, len(pca_features) // 2))
        inertias = []
        silhouette_scores = []
        
        print("Calculando métricas para diferentes valores de k...")
        
        for k in k_range:
            # K-means
            #kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans = KMeans(n_clusters=k, n_init='auto', algorithm="elkan")

            cluster_labels = kmeans.fit_predict(pca_features)
            
            # Métricas
            inertia = kmeans.inertia_
            silhouette = silhouette_score(pca_features, cluster_labels)
            
            inertias.append(inertia)
            silhouette_scores.append(silhouette)
            
            print(f"  k={k}: Silhouette={silhouette:.3f}, Inertia={inertia:.1f}")
        
        # Encontrar k óptimo (mejor silhouette score)
        optimal_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_range)[optimal_idx]
        best_silhouette = silhouette_scores[optimal_idx]
        
        print(f"\n🏆 Número óptimo de clusters: {optimal_k}")
        print(f"📊 Mejor Silhouette Score: {best_silhouette:.3f}")
        
        # Crear gráfico de análisis de clusters (sin t-SNE)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico del codo (Elbow method)
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Número de Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.set_title('📈 Método del Codo')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Óptimo k={optimal_k}')
        ax1.legend()
        
        # Silhouette score
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Número de Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('📊 Análisis Silhouette')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Óptimo k={optimal_k}')
        ax2.axhline(y=best_silhouette, color='green', linestyle=':', alpha=0.7, label=f'Score={best_silhouette:.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Guardar gráfico
        cluster_analysis_path = "cluster_analysis.png"
        plt.savefig(cluster_analysis_path, dpi=300, bbox_inches='tight')
        print(f"💾 Análisis de clusters guardado: {cluster_analysis_path}")
        
        plt.show()
        
        return optimal_k
    
    def ask_user_clusters(self, suggested_k):
        """
        Pregunta al usuario cuántos clusters quiere
        """
        print(f"\n🤔 SELECCIÓN DE CLUSTERS")
        print("=" * 30)
        print(f"💡 El análisis sugiere {suggested_k} clusters como óptimo")
        print("📊 Puedes ver los gráficos generados para tomar tu decisión")
        
        while True:
            try:
                user_input = input(f"\n¿Cuántos clusters quieres usar? (2-20, Enter para usar {suggested_k}): ").strip()
                
                if user_input == "":
                    return suggested_k
                
                k = int(user_input)
                if 2 <= k <= 20:
                    return k
                else:
                    print("❌ El número debe estar entre 2 y 20")
                    
            except ValueError:
                print("❌ Por favor ingresa un número válido")
    
    def perform_clustering(self, n_clusters):
        """
        Realiza el clustering final
        """
        print(f"\n🎯 CLUSTERING FINAL CON {n_clusters} CLUSTERS")
        print("=" * 50)
        
        if self.pca_features is None:
            print("❌ Primero ejecuta el análisis exploratorio")
            return None
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", algorithm="elkan")
        self.cluster_labels = kmeans.fit_predict(self.pca_features)
        
        # Calcular métricas
        silhouette = silhouette_score(self.pca_features, self.cluster_labels)
        
        # Estadísticas
        cluster_counts = Counter(self.cluster_labels)
        print(f"📊 Silhouette Score: {silhouette:.3f}")
        print(f"📊 Distribución de clusters:")
        
        for i in range(n_clusters):
            count = cluster_counts.get(i, 0)
            percentage = (count / len(self.cluster_labels)) * 100
            print(f"  🎯 Cluster {i}: {count:,} imágenes ({percentage:.1f}%)")
        
        # Mostrar visualización t-SNE con los clusters finales
        self._show_final_tsne_visualization(n_clusters)
        
        return kmeans
    
    def _show_final_tsne_visualization(self, n_clusters):
        """
        Muestra visualización t-SNE con los clusters finales seleccionados por el usuario
        """
        print("🎨 Generando visualización t-SNE final...")
        
        # Preparar datos para t-SNE (limitar puntos para eficiencia)
        max_samples = min(self.tsne_samples, len(self.pca_features))
        if len(self.pca_features) > self.tsne_samples:
            sample_indices = np.random.choice(len(self.pca_features), max_samples, replace=False)
            tsne_data = self.pca_features[sample_indices]
            tsne_labels = self.cluster_labels[sample_indices]
            print(f"   📊 Usando muestra de {max_samples} puntos de {len(self.pca_features)} total para visualización")
        else:
            tsne_data = self.pca_features
            tsne_labels = self.cluster_labels
            print(f"   📊 Usando todos los {len(self.pca_features)} puntos para visualización")
        
        try:
            # Aplicar t-SNE
            perplexity_value = min(30, len(tsne_data) - 1)
            tsne = TSNE(n_components=2,  perplexity=perplexity_value)
            tsne_result = tsne.fit_transform(tsne_data)
            
            # Crear figura para la visualización final
            plt.figure(figsize=(12, 8))
            
            # Crear scatter plot con colores por cluster
            colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
            
            for cluster_id in range(n_clusters):
                mask = tsne_labels == cluster_id
                if np.any(mask):
                    count = np.sum(mask)
                    plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                              c=[colors[cluster_id]], 
                              label=f'Cluster {cluster_id} ({count} imágenes)',
                              alpha=0.7, s=50)
            
            plt.xlabel('t-SNE Componente 1', fontsize=12)
            plt.ylabel('t-SNE Componente 2', fontsize=12)
            
            sample_size = len(tsne_data)
            total_images = len(self.cluster_labels)
            plt.title(f'🎪 Visualización Final de Clusters con t-SNE\n'
                     f'Clusters: {n_clusters} | Muestra: {sample_size} de {total_images} imágenes', 
                     fontsize=14, fontweight='bold')
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Guardar gráfico final
            final_tsne_path = "final_tsne_clusters.png"
            plt.savefig(final_tsne_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualización t-SNE final guardada: {final_tsne_path}")
            
            # Mostrar gráfico
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generando t-SNE final: {e}")
            print("   Continuando con el proceso...")
    
    def organize_into_folders(self, copy_files=True):
        """
        Organiza las imágenes en carpetas por cluster
        """
        print(f"\n🗂️ ORGANIZANDO IMÁGENES EN CARPETAS")
        print("=" * 40)
        
        if len(self.cluster_labels) == 0:
            print("❌ Primero ejecuta el clustering")
            return None, None
        
        # Crear carpeta principal de salida
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"Clustered_Images_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear carpetas para cada cluster
        n_clusters = len(set(self.cluster_labels))
        cluster_folders = {}
        
        for i in range(n_clusters):
            cluster_folder = os.path.join(output_dir, f"cluster_{i}")
            os.makedirs(cluster_folder, exist_ok=True)
            cluster_folders[i] = cluster_folder
            print(f"📁 Creada carpeta: cluster_{i}")
        
        # Organizar imágenes
        organized_count = 0
        cluster_stats = {i: {'count': 0, 'sources': Counter(), 'images': []} 
                        for i in range(n_clusters)}
        
        for idx, (img_path, img_name, folder_source, cluster_id) in enumerate(
            zip(self.image_paths, self.image_names, self.folder_labels, self.cluster_labels)):
            
            try:
                # Destino
                dest_folder = cluster_folders[cluster_id]
                
                # Crear nombre único para evitar conflictos
                base_name, ext = os.path.splitext(img_name)
                if folder_source != ".":
                    unique_name = f"{folder_source}_{base_name}{ext}"
                else:
                    unique_name = img_name
                dest_path = os.path.join(dest_folder, unique_name)
                
                # Copiar archivo
                if copy_files:
                    shutil.copy2(img_path, dest_path)
                else:
                    shutil.move(img_path, dest_path)
                
                # Actualizar estadísticas
                cluster_stats[cluster_id]['count'] += 1
                cluster_stats[cluster_id]['sources'][folder_source] += 1
                cluster_stats[cluster_id]['images'].append(unique_name)
                
                organized_count += 1
                
                if organized_count % 500 == 0:
                    print(f"  📋 Organizadas {organized_count:,} imágenes...")
                    
            except Exception as e:
                print(f"  ❌ Error organizando {img_name}: {e}")
                continue
        
        print(f"\n✅ Organizadas {organized_count:,} imágenes en {n_clusters} clusters")
        
        # Generar reporte
        self._generate_report(output_dir, cluster_stats)
        
        # Crear muestras visuales
        self._create_cluster_samples(output_dir, cluster_stats)
        
        return output_dir, cluster_stats
    
    def _generate_report(self, output_dir, cluster_stats):
        """
        Genera reporte detallado de los clusters
        """
        print("📊 Generando reporte...")
        
        report_path = os.path.join(output_dir, "clustering_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🗂️ REPORTE COMPLETO DE CLUSTERING DE IMÁGENES\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Carpeta de origen: {self.data_folder}\n")
            f.write(f"Total de imágenes procesadas: {sum(stats['count'] for stats in cluster_stats.values()):,}\n")
            f.write(f"Número de clusters: {len(cluster_stats)}\n")
            f.write(f"Características extraídas por imagen: {self.image_features.shape[1]}\n")
            f.write(f"Componentes PCA utilizados: {self.pca_features.shape[1]}\n\n")
            
            for cluster_id, stats in cluster_stats.items():
                f.write(f"🎯 CLUSTER {cluster_id}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Cantidad de imágenes: {stats['count']:,}\n")
                f.write(f"Porcentaje del total: {(stats['count'] / sum(s['count'] for s in cluster_stats.values())) * 100:.1f}%\n")
                f.write(f"Fuentes de datos:\n")
                
                for source, count in stats['sources'].items():
                    percentage = (count / stats['count']) * 100
                    f.write(f"  - {source}: {count:,} imágenes ({percentage:.1f}%)\n")
                
                f.write("\n")
        
        print(f"📄 Reporte guardado: {report_path}")
    
    def _create_cluster_samples(self, output_dir, cluster_stats, samples_per_cluster=30):
        """
        Crea muestras visuales de cada cluster
        """
        print("🖼️ Creando muestras visuales...")
        
        for cluster_id, stats in cluster_stats.items():
            if stats['count'] == 0:
                continue
                
            cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
            sample_images = stats['images'][:samples_per_cluster]
            
            # Crear visualización de muestra
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            fig.suptitle(f'Cluster {cluster_id} - Muestra de {len(sample_images)} imágenes\n'
                        f'Total: {stats["count"]} imágenes', 
                        fontsize=16, fontweight='bold')
            
            for i, ax in enumerate(axes.flat):
                if i < len(sample_images):
                    img_path = os.path.join(cluster_folder, sample_images[i])
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(sample_images[i][:25] + "..." if len(sample_images[i]) > 25 else sample_images[i], 
                                   fontsize=8)
                    except:
                        ax.text(0.5, 0.5, 'Error\ncargando\nimagen', 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.tight_layout()
            sample_path = os.path.join(output_dir, f"cluster_{cluster_id}_sample.png")
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  📸 Muestra del Cluster {cluster_id}: cluster_{cluster_id}_sample.png")

def main():
    """
    Función principal del script
    """
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Sistema completo de clustering de imágenes por características de color",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

ANÁLISIS COMPLETO:
  python complete_clustering.py Raw_data
  python complete_clustering.py "C:/Users/Usuario/MisDatos"
  python complete_clustering.py ./imagenes --max-images 1000

ANÁLISIS ÚNICAMENTE (sin organización):
  python complete_clustering.py Raw_data --only-analysis
  python complete_clustering.py Raw_data --only-analysis --max-images 500
  python complete_clustering.py Raw_data --only-analysis --tsne-samples 10000

El script realizará:

MODO COMPLETO:
1. Carga y análisis de características de color
2. Análisis directo del número óptimo de clusters
3. Visualización del método del codo y Silhouette
4. Clustering personalizado según tu elección
5. Organización automática en carpetas

MODO ANÁLISIS ÚNICAMENTE (--only-analysis):
1. Carga y análisis de características de color
2. Análisis de clusters óptimos (Codo + Silhouette)
3. Selección interactiva del número de clusters
4. Visualización t-SNE con clusters finales
5. NO organiza carpetas ni genera reportes
        """
    )
    
    parser.add_argument('data_folder', 
                       help='Ruta a la carpeta que contiene las imágenes o subcarpetas con imágenes')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Máximo número de imágenes por carpeta (opcional)')
    parser.add_argument('--no-copy', action='store_true',
                       help='Mover archivos en lugar de copiarlos')
    parser.add_argument('--only-analysis', action='store_true',
                       help='Solo realizar análisis y visualizaciones (sin organizar en carpetas)')
    parser.add_argument('--tsne-samples', type=int, default=5000,
                       help='Máximo número de puntos para visualización t-SNE (default: 5000)')
    
    args = parser.parse_args()
    
    # Verificar argumentos
    if not os.path.exists(args.data_folder):
        print(f"❌ Error: La carpeta '{args.data_folder}' no existe")
        sys.exit(1)
    
    # Verificar argumentos básicos
    # (No hay validaciones adicionales necesarias)
    
    if args.only_analysis:
        print("🎯 ANÁLISIS ÚNICAMENTE - SIN ORGANIZACIÓN")
        print("=" * 60)
        print("Este script analizará tus imágenes, mostrará el análisis de clusters")
        print("y generará solo las visualizaciones (sin organizar en carpetas).\n")
    else:
        print("🎯 SISTEMA COMPLETO DE CLUSTERING DE IMÁGENES")
        print("=" * 60)
        print("Este script analizará tus imágenes, mostrará el análisis de clusters")
        print("y te permitirá elegir el número de clusters óptimo.\n")
    
    try:
        # Crear instancia del sistema
        clustering_system = CompleteImageClustering(args.data_folder, tsne_samples=args.tsne_samples)
        
        # 1. Cargar imágenes
        clustering_system.load_images(max_images_per_folder=args.max_images)
        
        if len(clustering_system.image_features) == 0:
            print("❌ No se pudieron cargar imágenes")
            sys.exit(1)
        
        if args.only_analysis:
            # Modo de análisis únicamente (sin organización)
            # 2. Preparar datos y mostrar análisis de clusters
            suggested_k = clustering_system.prepare_data_for_analysis_only()
            
            # 3. Pregunta al usuario (igual que modo completo)
            final_k = clustering_system.ask_user_clusters(suggested_k)
            
            # 4. Clustering final
            kmeans_model = clustering_system.perform_clustering(final_k)
            
            if kmeans_model is None:
                print("❌ Error en el clustering")
                sys.exit(1)
                
            print(f"\n✅ ¡ANÁLISIS ÚNICAMENTE COMPLETADO!")
            print(f"📊 Análisis guardado: cluster_analysis.png")
            print(f"🎪 Visualización t-SNE guardada: final_tsne_clusters.png")
            print(f"ℹ️ Nota: No se organizaron carpetas (usa modo completo para eso)")
            
        else:
            # Modo completo original
            # 2. Preparar datos y análisis directo de clusters
            suggested_k = clustering_system.prepare_data_and_analyze_clusters()
            
            # 3. Pregunta al usuario
            final_k = clustering_system.ask_user_clusters(suggested_k)
            
            # 4. Clustering final
            kmeans_model = clustering_system.perform_clustering(final_k)
            
            if kmeans_model is None:
                print("❌ Error en el clustering")
                sys.exit(1)
            
            # 5. Organizar en carpetas
            output_dir, stats = clustering_system.organize_into_folders(copy_files=not args.no_copy)
            
            if output_dir:
                print(f"\n🎉 ¡CLUSTERING COMPLETADO EXITOSAMENTE!")
                print(f"📁 Resultados en: {output_dir}")
                print(f"📊 Consulta el reporte y las muestras visuales")
                
                # Mostrar resumen final
                print(f"\n📊 RESUMEN FINAL:")
                for cluster_id, cluster_stats in stats.items():
                    print(f"  🎯 Cluster {cluster_id}: {cluster_stats['count']:,} imágenes")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

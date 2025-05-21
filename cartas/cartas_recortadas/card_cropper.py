import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def crop_cards_grid(image_path, output_folder, rows=1, cols=4):
    """
    Recorta las cartas de una imagen basándose en una cuadrícula fija.
    
    Args:
        image_path: Ruta a la imagen con múltiples cartas
        output_folder: Carpeta donde se guardarán las cartas recortadas
        rows: Número de filas de cartas (default: 2)
        cols: Número de columnas de cartas (default: 4)
    
    Returns:
        count: Número de cartas recortadas
    """
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Leer la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return 0
    
    # Obtener dimensiones de la imagen
    height, width = img.shape[:2]
    
    # Calcular el tamaño de cada carta
    card_width = width // cols
    card_height = height // rows
    
    # Obtener el nombre base del archivo
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Recortar cada carta
    count = 0
    for r in range(rows):
        for c in range(cols):
            # Calcular las coordenadas para recortar
            left = c * card_width
            top = r * card_height
            right = left + card_width
            bottom = top + card_height
            
            # Recortar la carta
            card = img[top:bottom, left:right]
            
            # Guardar la carta recortada
            output_path = os.path.join(output_folder, f"{base_filename}_carta_{r+1}_{c+1}.jpg")
            cv2.imwrite(output_path, card)
            count += 1
    
    return count

def process_all_images(input_folder, output_folder, rows=1, cols=4):
    """
    Procesa todas las imágenes en la carpeta especificada.
    
    Args:
        input_folder: Carpeta con las imágenes a procesar
        output_folder: Carpeta donde se guardarán las cartas recortadas
        rows: Número de filas de cartas
        cols: Número de columnas de cartas
    """
    # Extensiones de imágenes comunes
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    total_cards = 0
    
    # Procesar cada archivo en la carpeta
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # Verificar si es un archivo de imagen
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in extensions):
            print(f"Procesando {filename}...")
            
            # Recortar cartas en cuadrícula
            cards_cropped = crop_cards_grid(file_path, output_folder, rows, cols)
            total_cards += cards_cropped
            
            print(f"  - {cards_cropped} cartas recortadas")
    
    print(f"\nProcesamiento completado. Total de cartas recortadas: {total_cards}")

# Función que permite visualizar la cuadrícula de recorte
def preview_grid(image_path, rows=1, cols=4):
    """
    Muestra una previsualización de cómo se recortará la imagen.
    
    Args:
        image_path: Ruta a la imagen con múltiples cartas
        rows: Número de filas de cartas
        cols: Número de columnas de cartas
    """
    # Leer la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return
    
    # Convertir de BGR a RGB para mostrar con matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Obtener dimensiones de la imagen
    height, width = img.shape[:2]
    
    # Calcular el tamaño de cada carta
    card_width = width // cols
    card_height = height // rows
    
    # Crear una copia para dibujar la cuadrícula
    img_grid = img_rgb.copy()
    
    # Dibujar líneas horizontales
    for r in range(1, rows):
        y = r * card_height
        cv2.line(img_grid, (0, y), (width, y), (0, 255, 0), 2)
    
    # Dibujar líneas verticales
    for c in range(1, cols):
        x = c * card_width
        cv2.line(img_grid, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Mostrar la imagen con la cuadrícula
    plt.figure(figsize=(10, 8))
    plt.imshow(img_grid)
    plt.title(f"Previsualización de la cuadrícula de recorte ({rows}x{cols})")
    plt.axis('off')
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Carpeta actual (donde está el script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Carpeta para guardar las cartas recortadas (dentro de la carpeta actual)
    output_dir = os.path.join(current_dir, "cartas_recortadas")
    
    # Configuración de la cuadrícula (2 filas, 4 columnas para la imagen de ejemplo)
    ROWS = 1
    COLS = 4
    
    # Procesar todas las imágenes en la carpeta actual
    process_all_images(current_dir, output_dir, ROWS, COLS)
    
    print(f"Las cartas recortadas están en: {output_dir}")
    
    # Opcional: Vista previa de la primera imagen encontrada
    for filename in os.listdir(current_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            preview_path = os.path.join(current_dir, filename)
            print(f"Mostrando previsualización para: {filename}")
            preview_grid(preview_path, ROWS, COLS)
            break
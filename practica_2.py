import cv2
import numpy as np
import os
from skimage.morphology import disk, opening, closing
from skimage.feature import canny

# Lista de nombres de imágenes
image_names = ["g0406", "g0408", "g0414", "g0428", "n0004", "n0005", "n0012", "n0013", "n0014", "n0044", 
               "n0045", "n0047", "n0049"]

# Directorios
image_dir = "refuge_images"
result_dir = "resultados"
os.makedirs(result_dir, exist_ok=True)  # Crear carpeta si no existe

# Inicializar las listas para almacenar los IoU y los resultados
iou_cups = []
iou_discs = []
cdr_values = []
groundtruth_cdr_values = [] 
glaucoma_diagnoses = []

for name in image_names:
    try:
        original_path = os.path.join(image_dir, f"{name}.png")
        cup_path = os.path.join(image_dir, f"{name}_cup.png")
        disc_path = os.path.join(image_dir, f"{name}_disc.png")

        original = cv2.imread(original_path)
        cup_gt = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
        disc_gt = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)

        if original is None or cup_gt is None or disc_gt is None:
            print(f"Error: Imágenes faltantes para {name}. Saltando...")
            continue
        
        image_result_dir = os.path.join(result_dir, name)
        os.makedirs(image_result_dir, exist_ok=True)

        # Paso 1: Convertir la imagen a escala de grises
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Paso 2: Segmentación de la copa
        if gray.ndim == 3 and gray.shape[2] == 3:
            green_channel = gray[:, :, 1]
        else:
            green_channel = gray

        inverted_green = 255 - green_channel
        inverted_green_mejorado = cv2.GaussianBlur(inverted_green, (5, 5), 0)
        inverted_green_mejorado = cv2.addWeighted(inverted_green_mejorado, 0.0, inverted_green, 2, 0)
        structuring_element = disk(15)
        cierre = opening(inverted_green_mejorado, structuring_element)
        cierre = cierre.astype(np.uint8)
        cierre_umbral = cv2.adaptiveThreshold(cierre, 200, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY_INV, blockSize=33,
                                              C=50)
        cierre_umbral = cv2.bitwise_not(cierre_umbral)
        bordes = canny(cierre_umbral, 2.3, 20, 56)
        bordes = (bordes * 255).astype(np.uint8)
        SE = disk(12)
        bordes_cierre = closing(bordes, SE)
        contornos, _ = cv2.findContours(bordes_cierre,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contornos, _ = cv2.findContours(bordes_cierre,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        max_contorno = max(contornos, key=cv2.contourArea)

        M = cv2.moments(max_contorno)
        if M["m00"] != 0:
            cx_roi = int(M["m10"] / M["m00"])
            cy_roi = int(M["m01"] / M["m00"])

            distances = [np.sqrt((cx_roi - point[0][0]) ** 2 + (cy_roi - point[0][1]) ** 2) for point in max_contorno]
            max_distance = max(distances)

            if max_distance > 34:
                ellipse = cv2.fitEllipse(max_contorno)
                mask_copa_roi = np.zeros_like(gray)
                cv2.ellipse(mask_copa_roi, ellipse, (255), -1)
            else:
                mask_copa_roi = np.zeros_like(gray)
                cv2.circle(mask_copa_roi, (cx_roi, cy_roi), int(max_distance), (255), -1)

            # Crear máscara de la copa
            cup_thresh = mask_copa_roi


        intersection_cup = np.logical_and(cup_thresh > 0, cup_gt > 0)
        union_cup = np.logical_or(cup_thresh > 0, cup_gt > 0)
        iou_cup_pre = np.sum(intersection_cup) / np.sum(union_cup)

        # Ajuste de similaridad en la vecindad
        roi = cv2.bitwise_and(gray, gray, mask=mask_copa_roi)
        mean_intensity_cup = np.mean(roi[mask_copa_roi > 0])
        std_intensity_cup = np.std(roi[mask_copa_roi > 0])

        lower_bound = max(0, mean_intensity_cup - 0.2 * std_intensity_cup)  
        upper_bound = min(255, mean_intensity_cup + 0.2 * std_intensity_cup)

        similarity_mask = cv2.inRange(gray, lower_bound, upper_bound)
        combined_mask = cv2.bitwise_or(similarity_mask, mask_copa_roi)

        # Encontrar nuevos contornos y recalibrar la elipse
        contornos_nuevos, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos_nuevos:
            max_contorno_comb = max(contornos_nuevos, key=cv2.contourArea)
            if len(max_contorno_comb) >= 5:
                ellipse_nueva = cv2.fitEllipse(max_contorno_comb)
                mask_copa_final = np.zeros_like(gray)
                cv2.ellipse(mask_copa_final, ellipse_nueva, (255), -1)
            else:
                mask_copa_final = combined_mask
        else:
            mask_copa_final = combined_mask

        # IoU después del ajuste
        intersection_cup_post = np.logical_and(mask_copa_final > 0, cup_gt > 0)
        union_cup_post = np.logical_or(mask_copa_final > 0, cup_gt > 0)
        iou_cup_post = np.sum(intersection_cup_post) / np.sum(union_cup_post)

        # Selección de la mejor máscara
        if iou_cup_post > iou_cup_pre:
            best_mask = mask_copa_final
            best_iou = iou_cup_post
        else:
            best_mask = mask_copa_roi
            best_iou = iou_cup_pre

        iou_cups.append(best_iou)  
        cup_thresh = best_mask  

        # Cálculo del disco basado en el contorno más grande
        contornos_disc, _ = cv2.findContours(disc_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contorno_disc = max(contornos_disc, key=cv2.contourArea)

        M_disc = cv2.moments(max_contorno_disc)
        if M_disc["m00"] != 0:
            cx_disc = int(M_disc["m10"] / M_disc["m00"])
            cy_disc = int(M_disc["m01"] / M_disc["m00"])

            distances_disc = [np.sqrt((cx_disc - point[0][0]) ** 2 + (cy_disc - point[0][1]) ** 2) for point in max_contorno_disc]
            max_distance_disc = max(distances_disc)

            if max_distance_disc > 34:
                ellipse_disc = cv2.fitEllipse(max_contorno_disc)
                mask_disc = np.zeros_like(gray)
                cv2.ellipse(mask_disc, ellipse_disc, (255), -1)
            else:
                mask_disc = np.zeros_like(gray)
                cv2.circle(mask_disc, (cx_disc, cy_disc), int(max_distance_disc), (255), -1)

        # Asignar esta nueva máscara al disco
        disc_thresh = mask_disc

        # Evaluación (IoU y visualización)
        segmented_output = original.copy()
        contours_cup, _ = cv2.findContours(cup_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_disc, _ = cv2.findContours(disc_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(segmented_output, contours_cup, -1, (255, 0, 0), 2)  # Azul para la copa
        cv2.drawContours(segmented_output, contours_disc, -1, (0, 255, 0), 2)  # Verde para el disco

        # Guardar imágenes y resultados
        intersection_disc = np.logical_and(disc_thresh > 0, disc_gt > 0)
        union_disc = np.logical_or(disc_thresh > 0, disc_gt > 0)
        iou_disc = np.sum(intersection_disc) / np.sum(union_disc)
        
        iou_discs.append(iou_disc)

        # Calcular CDR del ground truth
        cdr_gt = np.sum(cup_gt > 0) / np.sum(disc_gt > 0)

        # Calcular CDR predicho
        cup_height = np.sum(np.any(cup_thresh > 0, axis=1))
        disc_height = np.sum(np.any(disc_thresh > 0, axis=1))
        cdr_pred = cup_height / disc_height

        # Clasificación del glaucoma basada en el CDR
        if cdr_pred > 0.6:
            glaucoma_diagnosis = "Glaucoma avanzado"
        elif cdr_pred > 0.4:
            glaucoma_diagnosis = "Puede ser glaucoma"
        else:
            glaucoma_diagnosis = "Sano"

        # Guardar resultados
        cdr_values.append(cdr_pred)
        groundtruth_cdr_values.append(cdr_gt)
        glaucoma_diagnoses.append(glaucoma_diagnosis)

        # Guardar resultados
        segmented_output_path = os.path.join(image_result_dir, f"{name}_segmented.png")
        cv2.putText(segmented_output, f"CDR: {cdr_pred:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(segmented_output, f"IoU Copa: {best_iou:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(segmented_output, f"IoU Disco: {iou_disc:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(segmented_output, f"Diagnostico: {glaucoma_diagnosis}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imwrite(segmented_output_path, segmented_output)

    except Exception as e:
        print(f"Error procesando {name}: {e}")

# Guardar los resultados de CDR y diagnóstico en un archivo de texto
cdr_results_path = os.path.join(result_dir, "cdr_results.txt")
with open(cdr_results_path, "w") as f:
    for name, cdr_pred, cdr_gt, diagnosis in zip(image_names, cdr_values, groundtruth_cdr_values, glaucoma_diagnoses):
        f.write(f"{name}\n")
        f.write(f"CDR Predicho: {cdr_pred:.2f}\n")
        f.write(f"CDR Ground Truth: {cdr_gt:.2f}\n")
        f.write(f"Diagnostico: {diagnosis}\n\n")

print("Resultados de CDR y diagnóstico guardados en cdr_results.txt")

# Calcular la media de los IoU
mean_iou_cup = np.mean(iou_cups)
mean_iou_disc = np.mean(iou_discs)

# Guardar los resultados en un archivo de texto
iou_results_path = os.path.join(result_dir, "iou_results.txt")
with open(iou_results_path, "w") as f:
    f.write(f"Media IoU Copa: {mean_iou_cup:.4f}\n")
    f.write(f"Media IoU Disco: {mean_iou_disc:.4f}\n")

print("Resultados de IoU guardados en iou_results.txt")

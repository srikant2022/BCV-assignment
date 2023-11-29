import streamlit as st
import cv2
import numpy as np


def apply_translation(image):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 20]])  # Translation matrix
    translated_img = cv2.warpAffine(image, M, (cols, rows))
    return translated_img



def apply_scaling(image):
    scaled_img = cv2.resize(image, None, fx=0.5, fy=0.5)  # Scale by a factor of 0.5
    return scaled_img

def apply_shearing(image):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0.5, 0], [0.5, 1, 0]])  # Shearing matrix
    sheared_img = cv2.warpAffine(image, M, (cols, rows))
    return sheared_img


def apply_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def apply_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return corners


def convert_to_grayscale(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img



def enhance_color(image, alpha=1.0, beta=0):
    enhanced_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced_img


def apply_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# Corner Detection function
def apply_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.normalize(corners, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return corners

def main():
    st.title('OpenCV Image Processing App')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Geometric Transformations checkboxes
        geometric_operations = st.checkbox("Geometric Transformations")
        selected_geometric_operations = []
        if geometric_operations:
            translation = st.checkbox("Translation")
            
            scaling = st.checkbox("Scaling")
            shearing = st.checkbox("Shearing")
            flipping = st.checkbox("Flipping")

            if translation:
                selected_geometric_operations.append("Translation")
           
            if scaling:
                selected_geometric_operations.append("Scaling")
            if shearing:
                selected_geometric_operations.append("Shearing")
            if flipping:
                selected_geometric_operations.append("Flipping")

        # Color Transformations checkboxes
        color_operations = st.checkbox("Color Transformations")
        if color_operations:
            grayscale_conversion = st.checkbox("Grayscale Conversion")
           
            if color_space_conversion:
                selected_color_space = st.selectbox("Select Color Space", ("RGB", "HSV", "LAB", "CMYK"))
            color_enhancement = st.checkbox("Color Enhancement")
                
        
        # Edge Detection checkbox
        edge_detection = st.checkbox("Edge Detection")

        # Corner Detection checkbox
        corner_detection = st.checkbox("Corner Detection")

        # Geometric Transformations processing on button click
        if st.button("PROCESS") and (geometric_operations or color_operations):
            if selected_geometric_operations:
                st.header("Geometric Transformations")
                for operation in selected_geometric_operations:
                    if operation == 'Translation':
                        translated_img = apply_translation(image)
                        st.image(translated_img, caption='Translation', use_column_width=True)
                    elif operation == 'Rotation':
                        rotated_img = apply_rotation(image)
                        st.image(rotated_img, caption='Rotation', use_column_width=True)
                    elif operation == 'Scaling':
                        scaled_img = apply_scaling(image)
                        st.image(scaled_img, caption='Scaling', use_column_width=True)
                    elif operation == 'Shearing':
                        sheared_img = apply_shearing(image)
                        st.image(sheared_img, caption='Shearing', use_column_width=True)
                    elif operation == 'Flipping':
                        flipped_img = apply_flipping(image)
                        st.image(flipped_img, caption='Flipping', use_column_width=True)
            
            # Color Transformations
            if color_operations: 
                if grayscale_conversion:
                    grayscale_img = convert_to_grayscale(image)
                    st.header("Grayscale Conversion")
                    st.image(grayscale_img, caption='Grayscale Image', use_column_width=True)
                
                if color_space_conversion:
                    if selected_color_space:
                        color_space = getattr(cv2, f'COLOR_BGR2{selected_color_space}')
                        converted_img = convert_color_space(image, color_space)
                        st.header(f"Color Space Conversion to {selected_color_space}")
                        st.image(converted_img, caption=f"{selected_color_space} Image", use_column_width=True)

            # Edge Detection
            if edge_detection:
                edges = apply_edge_detection(image)
                st.header("Edge Detection")
                st.image(edges, caption='Edge Detected Image', use_column_width=True)

            # Corner Detection
            if corner_detection:
                corners = apply_corner_detection(image)
                st.header("Corner Detection")
                st.image(corners, caption='Corner Detected Image', use_column_width=True)
        
        if color_operations:    
            if color_enhancement:
                st.header("Color Enhancement")
                alpha = st.slider("Contrast (Alpha)", 0.0, 3.0, 1.0)
                beta = st.slider("Brightness (Beta)", 0, 100, 0)
                enhanced_img = enhance_color(image, alpha=alpha, beta=beta)
                st.image(enhanced_img, caption='Enhanced Image', use_column_width=True)
if __name__ == '__main__':
    main()
import os
import xml.etree.ElementTree as ET

# Function to convert a single XML file to YOLO format
def convert_xml_to_yolo(xml_file, output_dir, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    yolo_data = []

    for obj in root.findall('object'):
        obj_class = class_mapping.get(obj.find('name').text, -1)
        if obj_class == -1:
            continue

        robndbox = obj.find('robndbox')
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        x_center = cx / image_width
        y_center = cy / image_height
        width = w / image_width
        height = h / image_height

        yolo_format = f"{obj_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {angle:.6f}"
        yolo_data.append(yolo_format)

    output_file = os.path.join(output_dir, os.path.basename(xml_file).replace('.xml', '.txt'))
    with open(output_file, 'w') as f:
        f.write("\n".join(yolo_data))

# Function to process all XML files in a directory
def batch_convert_xml_to_yolo(input_dir, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(input_dir, xml_file), output_dir, class_mapping)

# Example usage
input_directory = './'
output_directory = 'txt_files'

# Define your class mapping here. Example: {'pig': 0, 'cow': 1}
class_mapping = {'pig': 0}

batch_convert_xml_to_yolo(input_directory, output_directory, class_mapping)

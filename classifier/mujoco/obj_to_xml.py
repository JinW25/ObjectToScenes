import argparse
import os
import xml.etree.ElementTree as ET

def modify_xml(xml_filepath):
    # Parse the XML file
    tree = ET.parse(xml_filepath)
    root = tree.getroot()

    # Find and remove the 'default' tag
    for default_tag in root.findall('default'):
        root.remove(default_tag)

    # Write the modified tree back to the file
    tree.write(xml_filepath)

parser = argparse.ArgumentParser(
    description="Remove the <default> block from per-object MuJoCo XML files "
                "(one sub-folder per object, e.g. as produced by obj2mjcf). Files are modified in place.")
parser.add_argument("merged_folder", type=str,
                    help="Folder containing one sub-folder per object with its .xml file")
args = parser.parse_args()
merged_folder = args.merged_folder

# For each object folder
for object_folder in os.listdir(merged_folder):
    object_folder_path = os.path.join(merged_folder, object_folder)
    if os.path.isdir(object_folder_path):
        # For each XML file inside the object folder
        for xml_file in os.listdir(object_folder_path):
            if xml_file.endswith('.xml'):
                print(f"xml_file: {xml_file}")
                modify_xml(os.path.join(object_folder_path, xml_file))

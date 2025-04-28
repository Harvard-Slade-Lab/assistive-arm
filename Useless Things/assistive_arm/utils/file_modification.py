from lxml import etree as ET

def modify_force_xml(xml_file, new_datafile):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for datafile in root.iter('datafile'):
        datafile.text = new_datafile

    with open(xml_file, 'wb') as file:
        file.write(ET.tostring(root, xml_declaration=True, encoding='utf-8'))
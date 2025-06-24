import xml.etree.ElementTree as ET


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    predicate_types = set()
    semantic_types = set()

    for predication in root.findall(".//Predication"):
        # Extract Predicate types
        predicate = predication.find("Predicate")
        if predicate is not None:
            predicate_types.add(predicate.get("type"))

        # Extract SemanticTypes from Subject
        subject = predication.find("Subject")
        if subject is not None:
            for sem_type in subject.findall(".//SemanticType"):
                semantic_types.add(sem_type.text)

        # Extract SemanticTypes from Object
        obj = predication.find("Object")
        if obj is not None:
            for sem_type in obj.findall(".//SemanticType"):
                semantic_types.add(sem_type.text)

    return list(predicate_types), list(semantic_types)


# Example usage
xml_file = "../COMBINI-data/adjudicated.xml"  # Replace with your actual XML file path
predicate_types, semantic_types = parse_xml(xml_file)

print("Unique Predicate Types:", predicate_types)
print("Number of predicate types:", len(predicate_types))
print("Unique Semantic Types:", semantic_types)

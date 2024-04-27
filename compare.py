import json

def compare_patient_ids(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    patient_ids1 = [item['patient_id'] for item in data1]
    patient_ids2 = [item['patient_id'] for item in data2]

    if patient_ids1 == patient_ids2:
        print("The patient IDs in the JSON files are identical.")
    else:
        print("The patient IDs in the JSON files have differences:")
        for i, (id1, id2) in enumerate(zip(patient_ids1, patient_ids2)):
            if id1 != id2:
                print(f"Mismatch at index {i}:")
                print(f"File 1: {id1}")
                print(f"File 2: {id2}")
                print()

# Usage example
compare_patient_ids('25389_25285_assignment2.json', 'assignment2_updated.json')
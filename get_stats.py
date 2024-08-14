import os
import csv
import re

def parse_stats(stats_text):
    parts = stats_text.split('|')
    fovs = int(re.search(r'\d+', parts[0]).group())
    rbcs = int(re.search(r'([\d,]+)', parts[1]).group().replace(',', ''))
    positives = int(re.search(r'([\d,]+)', parts[2]).group().replace(',', ''))
    #parasites_per_ul = int(re.search(r'([\d,]+)', parts[3]).group().replace(',', ''))
    parasites_per_ul = positives * ( 5000000 / rbcs )
    
    # Recalculate parasitemia as a fraction
    parasitemia = positives / rbcs if rbcs > 0 else 0
    
    return {
        'FOVs': fovs,
        'RBCs': rbcs,
        'Positives': positives,
        'Parasites_per_ul': parasites_per_ul,
        'Parasitemia': parasitemia
    }

def process_patients_folder(main_folder):
    results = []
    
    for patient_folder in os.listdir(main_folder):
        patient_path = os.path.join(main_folder, patient_folder)
        if os.path.isdir(patient_path):
            stats_file = os.path.join(patient_path, 'stats.txt')
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats_text = f.read().strip()
                    stats = parse_stats(stats_text)
                    stats['Patient_ID'] = patient_folder
                    results.append(stats)
    
    return results

def save_to_csv(results, output_file):
    fieldnames = ['Patient_ID', 'FOVs', 'RBCs', 'Positives', 'Parasites_per_ul', 'Parasitemia']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

# Main execution
if __name__ == "__main__":
    main_folder = "/media/lin/Extreme SSD/SBC20240725"  # Replace with the path to your main folder
    output_csv = "patient_stats_summary.csv"  # Name of the output CSV file
    
    results = process_patients_folder(main_folder)
    save_to_csv(results, output_csv)
    
    print(f"Stats for {len(results)} patients have been compiled into {output_csv}")
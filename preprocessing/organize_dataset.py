import os
import shutil


# ==========================================
# SOURCE
# ==========================================

SOURCE_DIR = r"C:\Users\darsh\OneDrive\Desktop\MINI-PROJECT\BreastCancer_XAI\400X"


# ==========================================
# DESTINATION
# ==========================================

DEST_DIR = r"C:\Users\darsh\OneDrive\Desktop\MINI-PROJECT\BreastCancer_XAI\dataset"


# ==========================================
# CLASS MAPPING
# ==========================================

CLASS_MAPPING = {
    "_A-": "adenosis",
    "_F-": "fibroadenoma",
    "_PT-": "phyllodes_tumor",
    "_TA-": "tubular_adenoma",
    "_DC-": "ductal_carcinoma",
    "_LC-": "lobular_carcinoma",
    "_MC-": "mucinous_carcinoma",
    "_PC-": "papillary_carcinoma"
}


# ==========================================
# CREATE CLASS FOLDERS
# ==========================================

for class_name in CLASS_MAPPING.values():

    os.makedirs(
        os.path.join(DEST_DIR, class_name),
        exist_ok=True
    )


# ==========================================
# PROCESS ALL NESTED FOLDERS
# ==========================================

total = 0
unknown = 0


print("\n======================================")
print("ORGANIZING DATASET")
print("======================================\n")


for root, dirs, files in os.walk(SOURCE_DIR):

    for filename in files:

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        ):
            continue


        source_file = os.path.join(
            root,
            filename
        )


        filename_upper = filename.upper()

        detected_class = None


        # Find class from filename
        for code, class_name in CLASS_MAPPING.items():

            if code in filename_upper:

                detected_class = class_name
                break


        # Unknown class
        if detected_class is None:

            print("UNKNOWN:", filename)

            unknown += 1

            continue


        # Destination
        destination_folder = os.path.join(
            DEST_DIR,
            detected_class
        )


        destination_file = os.path.join(
            destination_folder,
            filename
        )


        # Avoid duplicate filename
        if os.path.exists(destination_file):

            name, extension = os.path.splitext(filename)

            counter = 1

            while os.path.exists(destination_file):

                new_name = f"{name}_{counter}{extension}"

                destination_file = os.path.join(
                    destination_folder,
                    new_name
                )

                counter += 1


        shutil.copy2(
            source_file,
            destination_file
        )


        total += 1


# ==========================================
# RESULT
# ==========================================

print("\n======================================")
print("DATASET ORGANIZATION COMPLETED")
print("======================================")

print(
    f"\nTotal images organized: {total}"
)

print(
    f"Unknown images: {unknown}"
)


# ==========================================
# CLASS COUNTS
# ==========================================

print("\nClass-wise image count:")
print("--------------------------------------")


for class_name in CLASS_MAPPING.values():

    folder = os.path.join(
        DEST_DIR,
        class_name
    )

    count = 0

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".gif")
            ):

                count += 1


    print(
        f"{class_name:20s}: {count}"
    )


print("\n======================================")
print("DONE")
print("======================================")
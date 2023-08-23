import os
import LoadImages
import PyPDF2

def extract_image_from_pdf(pdf_path, output_folder, file_num):
    pdf = PyPDF2.PdfReader(pdf_path)
    image_count = 0

    for page_num, page in enumerate(pdf.pages):
        xObject = page['/Resources']['/XObject'].get_object()
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                image_count += 1
                if image_count % 2 == 0: continue
                img = xObject[obj].get_data()
                img_file_path = os.path.join(output_folder, f"image_{file_num}.png")
                with open(img_file_path, 'wb') as f:
                    f.write(img)

def extract_images_from_pdfs(pdf_folder, output_folder):

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_num, pdf_filename in enumerate(os.listdir(pdf_folder)):
        if pdf_filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_filename)
            extract_image_from_pdf(pdf_path, output_folder, file_num)

# Main script
if __name__ == "__main__":
    pdf_folder = "Files"
    output_folder = "output_images"
    chunk_size = 25

    extract_images_from_pdfs(pdf_folder, output_folder)
    LoadImages.split_images(output_folder, chunk_size)
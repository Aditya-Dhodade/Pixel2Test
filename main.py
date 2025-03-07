import streamlit as st
from PIL import Image
from groq import Groq
from image_utils import convert_image_to_base64, display_base64_image
import base64
import json
import re
import subprocess
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# import tensorflow.lite as tflite

# interpreter = tflite.Interpreter(model_path="model.tflite", experimental_delegates=[])
# interpreter.allocate_tensors()

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

# Path to your ChromeDriver
chrome_driver_path = "D:/chromedriver-win64/chromedriver-win64/chromedriver.exe"

# Initialize the WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)


if 'test_cases' not in st.session_state:
    st.session_state.test_cases = None
if 'selected_test_case' not in st.session_state:
    st.session_state.selected_test_case = None
if 'website_url' not in st.session_state:
    st.session_state.website_url = ""
if 'feature_file' not in st.session_state:
    st.session_state.feature_file = ""
if 'step_definition' not in st.session_state:
    st.session_state.step_definition = ""
if 'edited_code' not in st.session_state:
    st.session_state.edited_code = ""
if 'run_tests' not in st.session_state:
    st.session_state.run_tests = False
if 'extracted_xpath' not in st.session_state:
    st.session_state.extracted_xpath = {}
if 'num_test_cases' not in st.session_state:
    st.session_state.extracted_xpath = 0
if 'html_table' not in st.session_state:
    st.session_state.html_table = None
if 'testcase_done' not in st.session_state:
    st.session_state.testcase_done = True
if 'response' not in st.session_state:
    st.session_state.response = ""
# Initialize GROQ client
client = Groq(api_key="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")

PROJECT_DIR = r"D:/Sarvatra/integrated/v1/v2tollama"  # Change to the correct path

FEATURE_FILE_PATH = os.path.join(PROJECT_DIR, "features")  # Path where feature files are stored
STEP_PATH = os.path.join(PROJECT_DIR, "features/steps")

num = -1
st.set_page_config(page_title="Make Test cases using GenAI", page_icon=":Target:")
st.session_state.joining_option = None

def parse_response(response):
    """
    Parses the response into feature file and step definition parts.
    """
    feature_file_content = ""
    step_definition_content = ""

    # Look for the marker explicitly
    if "python" in response:
        parts = response.split("python", 1)
        feature_file_content = parts[0].strip()
        step_definition_content = parts[1].strip()
    else:
        # Fallback if the marker isn't present
        feature_file_content = response.strip()
        step_definition_content = "Python step definition not generated or incorrectly formatted."

    return feature_file_content, step_definition_content

def clean_python_code(content):
    """
    Removes unnecessary lines from Python code.

    Args:
        content (str): The full content of the Python file.

    Returns:
        str: Cleaned Python code without unnecessary lines.
    """
    # Split the content into lines
    print("clean python called")
    lines = content.splitlines()

    # Remove lines starting from a specific phrase
    clean_lines = []
    for line in lines:
        if line.startswith("```"):
            break  # Stop processing when encountering the unnecessary section
        clean_lines.append(line)

    return "\n".join(clean_lines)


def extract_code(content, language):
    """
    Extracts only code content based on the language syntax.

    Args:
        content (str): The input text containing code and other text.
        language (str): The programming language or file type (e.g., 'gherkin', 'python').

    Returns:
        str: Cleaned content containing only the code.
    """
    if language == "gherkin":
        # Extract lines starting with Gherkin keywords or scenarios
        keywords = ["Feature:", "Scenario:", "Given", "When", "Then", "And I", "But"]
        code_lines = [line for line in content.splitlines() if any(line.strip().startswith(k) for k in keywords)]
    elif language == "python":
        # Extract Python code (non-empty lines, ignoring comments and extra text)
        code_lines = [line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    else:
        # Default to returning all content
        code_lines = content.splitlines()

    return "\n".join(code_lines)

# converstion image to base64
def image_to_base64(image_path):

    """Converts an image file to base64 encoding."""

    with open(image_path, "rb") as image_file:

        return base64.b64encode(image_file.read()).decode('utf-8')
# Function to upload images
def upload_image():
    st.session_state.uploaded_images = []
    st.markdown(
        "<span style='font-size: 20px;'>Upload the UI images to generate test cases</span>",
        unsafe_allow_html=True,
    )
    
    # File uploader component
    images = st.file_uploader(
        "", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    
    # Store uploaded images in session state
    if images:
        for image in images:
            img = Image.open(image)
            st.session_state.uploaded_images.append(img)
        if len(st.session_state.uploaded_images) > 1:
            st.session_state.joining_option = st.selectbox("Select joining option:", ["select the joining method", "Horizontal", "Vertical"])
        
        # Display the uploaded images
        cols = st.columns(len(st.session_state.uploaded_images))
        for i, col in enumerate(cols):
            col.markdown(f"Image {i + 1}")
            col.image(st.session_state.uploaded_images[i])
        st.markdown("---")

# Function to join images
def join_images(images, orientation):
    print("joining images")
    print(orientation)
    if orientation == "Horizontal":
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        joined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            joined_image.paste(img, (x_offset, 0))
            x_offset += img.width
        joined_image.save("join.png")
        print('image saved')
    elif orientation == "Vertical":
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        joined_image = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in images:
            joined_image.paste(img, (0, y_offset))
            y_offset += img.height
        joined_image.save("join.png")
        print("image saved")

    print("image joining completed")
    return joined_image

# Initialize session state for uploaded images
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

# Function to generate HTML table from JSON response
def generate_table(response):
    try:
        print("table generation")
        # Use the Groq client to call the model and generate an HTML table from JSON
        prompt = (
            "Convert the following information into table format. Provide the code between "
            "<table> </table> only. The information is: " + response
        )
        
        client = Groq(api_key='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')
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        response_text = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            # print(chunk.choices[0].delta.content, end="")
                response_text += str(chunk.choices[0].delta.content)
        start_index = response_text.find("<table>")
        end_index = response_text.find("</table>") + 8
        print(str(response_text[start_index:end_index]))
        return str(response_text[start_index:end_index])

    except Exception as e:
        return str(e)

# Function to run the model
@st.cache_data(show_spinner=False)
def run_language_model(query, image_base_64):
    image_path = "join.png"
    image_base64 = image_to_base64(image_path)
    response1 = ""
    client = Groq(api_key='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')
    
    # Call the Llama 3.2 Vision model
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    # Process and print the response
    print("processing response")
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            # print(chunk.choices[0].delta.content, end="")
            response1 += str(chunk.choices[0].delta.content)
    print("----test case generated-------------")
    print(response1)
    
    return response1

# Upload image section
upload_image()

# Add the 'Done' button to trigger model execution
# Function to parse test cases from JSON response

def extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def parse_test_cases(response):
    import json
    import re

    try:
        # Extract the JSON part from the response
        json_data = extract_json(response)
        if json_data:
            return json.loads(json_data)
        else:
            return "No valid JSON found in response."
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {str(e)}"


# Function to display information of a single test case
def display_test_case(test_cases, selected_id):
    try:
        for test_case in test_cases:
            if test_case.get('Test Case ID') == selected_id:
                return test_case
        return f"No test case found with ID: {selected_id}"
    except Exception as e:
        return str(e)


# Function to extract XPaths from a website
# Function to extract elements with meaningful labels
def extract_xpaths(url):
    driver.get(url)
    time.sleep(3)  # Wait for the page to load

    # List of interactive element tags
    interactive_tags = ["input", "button", "select", "textarea", "a", "p", "span"]

    # Dictionary to store element types, labels, and XPaths
    interactive_elements = {}

    for tag in interactive_tags:
        elements = driver.find_elements(By.TAG_NAME, tag)
        for element in elements:
            if element.is_displayed():
                xpath = get_xpath(element)
                label = get_element_label(element, tag)
                if(label != "Unknown" and len(label)<=25):
                    element_type = element.get_attribute("type") if tag == "input" else tag
                
                    if element_type not in interactive_elements:
                        interactive_elements[element_type] = []
                    
                    interactive_elements[element_type].append({"label": label, "xpath": xpath})
                
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", interactive_elements)
    return interactive_elements

# Function to get the best label for an element
def get_element_label(element, tag):
    """Extract the best label for an element."""
    if tag == "a":
        # For anchor tags, use the text content as the label
        label = element.text.strip()
        return label if label else "Unknown"

    label = (element.get_attribute("name") or 
             element.get_attribute("placeholder") or 
             element.get_attribute("title") or
             element.get_attribute("aria-label") or 
             element.text.strip())

    if not label:
        # Try finding an associated <label> tag
        try:
            label_element = element.find_element(By.XPATH, f"//label[@for='{element.get_attribute('id')}']")
            label = label_element.text.strip()
        except:
            label = "Unknown"

    return label


# Function to generate XPath for an element
def get_xpath(element):
    script = """
    function getElementXPath(element) {
        if (!element) return "";
        let path = [];
        while (element.nodeType === Node.ELEMENT_NODE) {
            let index = 1, sibling = element.previousSibling;
            while (sibling) {
                if (sibling.nodeType === Node.ELEMENT_NODE && sibling.tagName === element.tagName) index++;
                sibling = sibling.previousSibling;
            }
            path.unshift(element.tagName.toLowerCase() + "[" + index + "]");
            element = element.parentNode;
        }
        return "/" + path.join("/");
    }
    return getElementXPath(arguments[0]);
    """
    return driver.execute_script(script, element)

# Function to generate feature and step definition files
def generate_feature_and_step_definition(test_case, selected_id, xpaths=None):
    print("its called")
    prompt = f"""
    {test_case}
    Above given is a test case scenario based on a form. Generate the feature file corresponding to the given test case and its Python step definition script.
    This is the website for testing {st.session_state.website_url} Use this website in the step definition file """
    if xpaths:
        prompt += f"\n\nHere are the XPaths of the elements on the website:\n{xpaths}"

    # GROQ API call
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an AI assistant that generates BDD feature files and Python step definition scripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        # Extract the result
        response = completion.choices[0].message.content.strip()
        # print("--------------------------")
        # print(response)
        # Parse the response
        feature_file_content, step_definition_content = parse_response(response)
        # print("--------------------------")
        # print(feature_file_content)
        # print("--------------------------")
        # print(step_definition_content)
        # Extract only the code for feature file and step definition
        cleaned_feature_file = extract_code(feature_file_content, "gherkin")
        cleaned_step_definition = clean_python_code(step_definition_content)
        # print("--------------------------")
        # print(cleaned_feature_file)
        # print("--------------------------")
        # print(cleaned_step_definition)
        # feature_filename = f"{selected_id}_feature_file.feature"
        # step_filename = f"{selected_id}_step_definition.py"
        # # Save feature file and step definition for execution
        # with open(os.path.join(FEATURE_FILE_PATH, feature_filename), "w") as f:
        #     f.write(cleaned_feature_file)
        # with open(os.path.join(STEP_PATH, step_filename), "w") as f:
        #     f.write(cleaned_step_definition)

        return cleaned_feature_file, cleaned_step_definition
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None, None


# taking input of website
st.session_state.website_url = st.text_input("Enter the website URL (Environement):")
if st.session_state.website_url == "None":
    print("None-----------")
    st.session_state.num_test_cases = st.number_input("Enter the number of test cases to generate:", step=1)

    if st.session_state.num_test_cases != 0:
        with st.expander("Generate testcases"):
        # Ensure images are uploaded before continuing
            if st.session_state.uploaded_images:
                num_images = len(st.session_state.uploaded_images)
                joined_image = None
                if num_images > 1:
                    # Join images based on user selection
                    joined_image = join_images(st.session_state.uploaded_images, st.session_state.joining_option)
                    joined_image.save("join.png")
                else:
                    joined_image = st.session_state.uploaded_images[0]
                    joined_image.save("join.png")

                # Generate the appropriate question
                if num_images > 1:
                    question = (
                        f"Write {st.session_state.num_test_cases} QA 'Test cases' in JSON format for this joined image sequence. "
                        "The images are joined either horizontally or vertically with arrows indicating flow or sequence between them. "
                        "For each test case, use the following keys: "
                        "'Test Case ID', 'Description of the Test Case', 'Steps to Perform the Test Case', and 'Expected Result'. "
                        "Analyze both images in sequence, ensuring each test case covers elements or interactions present in both images, "
                        "reflecting any form fields, buttons, or UI components that change between images. "
                        "Include updates, dependencies, or transitions between the images where relevant."
                    )
                else:
                    question = (
                        f"Write {st.session_state.num_test_cases} QA 'Test cases' in valid JSON array format for this image. "
                        "Use these keys for each test case: 'Test Case ID', 'Description of the Test Case', "
                        "'Steps to Perform the Test Case', 'Expected Result'. "
                        "Ensure the JSON starts with '[' and ends with ']', and has no extra text."
                    )

                if st.session_state.joining_option != "select the joining method" or num_images == 1:
                    # Custom spinner
                    with st.spinner("Generating test cases..."):
                        with st.container():
                            # Convert image to base64
                            image_b64 = convert_image_to_base64("join.png")

                            # Run the model with the user's question
                            response = run_language_model(question, image_b64)
                            print("Raw Response:", response)

                            # Parse JSON test cases
                            st.session_state.test_cases = parse_test_cases(response)
                            # Generate HTML table
                            html_table = generate_table(response)

                            # Display the table
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            st.markdown(
                                f"""
                                <div style="background-color: #262730; padding: 16px; border-radius: 8px; max-width: 5000px; color: white;">
                                    <h3 style="margin-top: 0;">Response:</h3>
                                    <p style="font-size: 16px;">{html_table}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if isinstance(st.session_state.test_cases, list):
                                # Dropdown for selecting a test case ID
                                test_case_ids = [tc.get('Test Case ID', f"Unknown {i+1}") for i, tc in enumerate(st.session_state.test_cases)]
                                selected_id = st.selectbox("Select Test Case ID to Extract:", test_case_ids)

                                # Button to extract selected test case information
                                if st.button("Extract Test Case"):
                                    print("extracting test case")
                                    st.session_state.selected_test_case = display_test_case(st.session_state.test_cases, selected_id)
                                    st.write(st.session_state.selected_test_case)

            

                                # Button to generate feature file and step definition
                                if st.button("Generate Feature File and Step Definition"):

                                    if st.session_state.selected_test_case:
                                
                                            with st.spinner("Generating feature and step definition files..."):
                                                # Pass elements instead of xpaths
                                                st.session_state.feature_file, st.session_state.step_definition = generate_feature_and_step_definition(st.session_state.selected_test_case, selected_id)
                                                
                                                if st.session_state.feature_file and st.session_state.step_definition:
                                                    st.subheader("Generated Feature File (Cleaned)")
                                                    st.code(st.session_state.feature_file, language="gherkin")

                                                    st.subheader("Generated Python Step Definition Script (Cleaned)")
                                                    st.code(st.session_state.step_definition, language="python")
                                                    

                
                                    else:
                                        st.error("No test case selected. Please extract a test case first.")
                            else:
                                st.error("Failed to parse test cases. Ensure the response is in JSON format.")
                else:
                    st.error("Please upload at least one image before pressing 'Done'.")
                
elif st.session_state.website_url == "":
    pass
else:
    print(st.session_state.website_url)
    with st.spinner("Extracting xpath from "  + str(st.session_state.website_url)):
        st.session_state.extracted_xpath = extract_xpaths(st.session_state.website_url)
    if st.session_state.extracted_xpath != {}:
        st.success("XPaths extracted successfully!")
    st.write(st.session_state.extracted_xpath)
    st.session_state.num_test_cases = st.number_input("Enter the number of test cases to generate:",  step=1)
    if st.session_state.num_test_cases != 0:
        with st.expander("Generate testcases"):
        # Ensure images are uploaded before continuing
            if st.session_state.uploaded_images:
                num_images = len(st.session_state.uploaded_images)
                joined_image = None
                if num_images > 1:
                    # Join images based on user selection
                    joined_image = join_images(st.session_state.uploaded_images, st.session_state.joining_option)
                    joined_image.save("join.png")
                else:
                    joined_image = st.session_state.uploaded_images[0]
                    joined_image.save("join.png")

                # Generate the appropriate question
                if num_images > 1:
                    question = (
                        f"Write {st.session_state.num_test_cases} QA 'Test cases' in JSON format for this joined image sequence. "
                        "The images are joined either horizontally or vertically with arrows indicating flow or sequence between them. "
                        "For each test case, use the following keys: "
                        "'Test Case ID', 'Description of the Test Case', 'Steps to Perform the Test Case', and 'Expected Result'. "
                        "Analyze both images in sequence, ensuring each test case covers elements or interactions present in both images, "
                        "reflecting any form fields, buttons, or UI components that change between images. "
                        "Include updates, dependencies, or transitions between the images where relevant."
                    )
                else:
                    question = (
                        f"Write {st.session_state.num_test_cases} QA 'Test cases' in valid JSON array format for this image. "
                        "Use these keys for each test case: 'Test Case ID', 'Description of the Test Case', "
                        "'Steps to Perform the Test Case', 'Expected Result'. "
                        "Ensure the JSON starts with '[' and ends with ']', and has no extra text."
                    )

                if st.session_state.joining_option != "select the joining method" or num_images == 1:
                    # Custom spinner
                    
                    with st.spinner("Generating test cases..."):
                        with st.container():
                            # Convert image to base64
                            image_b64 = convert_image_to_base64("join.png")

                            # Run the model with the user's question
                            response = run_language_model(question, image_b64)
                            print("Raw Response:", response)

                            # Parse JSON test cases
                            st.session_state.test_cases = parse_test_cases(response)
                            # Generate HTML table
                            st.session_state.html_table = generate_table(response)

                            # Display the table
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            st.markdown(
                                f"""
                                <div style="background-color: #262730; padding: 16px; border-radius: 8px; max-width: 5000px; color: white;">
                                    <h3 style="margin-top: 0;">Response:</h3>
                                    <p style="font-size: 16px;">{st.session_state.html_table}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # feedback code
                            # feedback_counter = 0
                            # while(st.session_state.testcase_done):
                            #     feedback = st.text_input("Feedback on the testcase", key=f"feedback_{feedback_counter}")
                            #     print(feedback_counter)
                            #     feedback_counter += 1
                            #     prompt1 = "This is my current response give by model. The response is {st.session_state.response}. But there are some suggestion by the user on the response. The suggestion is {feedback}. Now i want you to analyse the image and make the testcase by taking care of suggestion as well as the previous respsone. Use these keys for each test case: 'Test Case ID', 'Description of the Test Case', 'Steps to Perform the Test Case', 'Expected Result'. Ensure the JSON starts with '[' and ends with ']', and has no extra text"
                            #     image_b64 = convert_image_to_base64("join.png")

                            #     # Run the model with the user's question
                            #     response = run_language_model(prompt1, image_b64)
                            #     st.session_state.html_table = generate_table(response)
                            #     if st.button("Done"):
                            #         st.session_state.testcase_done = False

                            if isinstance(st.session_state.test_cases, list):
                                # Dropdown for selecting a test case ID
                                test_case_ids = [tc.get('Test Case ID', f"Unknown {i+1}") for i, tc in enumerate(st.session_state.test_cases)]
                                selected_id = st.selectbox("Select Test Case ID to Extract:", test_case_ids)
                                st.session_state.selected_test_case = display_test_case(st.session_state.test_cases, selected_id)
                                # # Button to extract selected test case information
                                # if st.button("Extract Test Case"):
                                #     print("extracting test case")
                                #     st.session_state.selected_test_case = display_test_case(st.session_state.test_cases, selected_id)
                                # if st.session_state.selected_test_case != None:
                                #     st.write(st.session_state.selected_test_case)


                                # Button to generate feature file and step definition
                                if st.button("Generate Feature File and Step Definition"):

                                    if st.session_state.selected_test_case:
                                            with st.spinner("Generating feature and step definition files..."):
                                                # Pass elements instead of xpaths
                                                st.session_state.feature_file, st.session_state.step_definition = generate_feature_and_step_definition(st.session_state.selected_test_case, selected_id, st.session_state.extracted_xpath)
                                                
                                                # if feature_file and step_definition:
                                                #     st.subheader("Generated Feature File (Cleaned)")
                                                #     # st.code(feature_file, language="gherkin")

                                                #     st.subheader("Generated Python Step Definition Script (Cleaned)")
                                                #     # st.code(step_definition, language="python")

                                    else:
                                        st.error("No test case selected. Please extract a test case first.")
                                if st.session_state.feature_file != "" and st.session_state.step_definition != "":
                                    st.subheader("Generated Feature File (Cleaned)")
                                    st.code(st.session_state.feature_file, language="gherkin")
                                    st.subheader("Generated Python Step Definition Script (Cleaned)")
                                    st.session_state.edited_code = st.text_area(
                                        "Edit the code below:",
                                        value=st.session_state.step_definition,
                                        height=400,
                                        key="code_editor"
                                    )

                                    # Display the edited code with syntax highlighting
                                    st.code(st.session_state.edited_code, language="python")
                                if st.button("Run Behave Tests"):
                                                    # st.session_state.run_tests = True
                                                    print("BEHAVE")
                                                    with st.spinner("Running tests..."):
                                                        try:
                                                            feature_filename = f"{selected_id}_feature_file.feature"
                                                            step_filename = f"{selected_id}_step_definition.py"
                                                            # Save feature file and step definition for execution
                                                            with open(os.path.join(FEATURE_FILE_PATH, feature_filename), "w") as f:
                                                                f.write(st.session_state.feature_file)
                                                            with open(os.path.join(STEP_PATH, step_filename), "w") as f:
                                                                f.write(st.session_state.edited_code)
                                                            # Open VS Code in the project directory
                                                            subprocess.Popen(["code", PROJECT_DIR], shell=True)

                                                            # Run Behave tests using subprocess
                                                            result = subprocess.run([r"D:/Sarvatra/integrated/v1/v2tollama/venv/Scripts/python.exe", "-m", "behave"], cwd=FEATURE_FILE_PATH, capture_output=True, text=True)
                                                            print(result, "result")
                                                            # Display output in Streamlit UI
                                                            if result.returncode == 0:
                                                                st.success("Tests executed successfully! ✅")
                                                                st.text_area("Test Output:", result.stdout)
                                                            else:
                                                                st.error("Test execution failed! ❌")
                                                                st.text_area("Error Output:", result.stderr)
                                                        except Exception as e:
                                                            st.error(f"An error occurred: {e}")
                            else:
                                st.error("Failed to parse test cases. Ensure the response is in JSON format.")
                else:
                    st.error("Please upload at least one image before pressing 'Done'.")


    
# Streamlit UI
# with st.expander("Generate testcases"):
#     # Ensure images are uploaded before continuing
#     if st.session_state.uploaded_images:
#         num_images = len(st.session_state.uploaded_images)
#         joined_image = None
#         if num_images > 1:
#             # Join images based on user selection
#             joined_image = join_images(st.session_state.uploaded_images, st.session_state.joining_option)
#             joined_image.save("join.png")
#         else:
#             joined_image = st.session_state.uploaded_images[0]
#             joined_image.save("join.png")

#         # Generate the appropriate question
#         if num_images > 1:
#             question = (
#                 "Write 5 QA 'Test cases' in JSON format for this joined image sequence. "
#                 "The images are joined either horizontally or vertically with arrows indicating flow or sequence between them. "
#                 "For each test case, use the following keys: "
#                 "'Test Case ID', 'Description of the Test Case', 'Steps to Perform the Test Case', and 'Expected Result'. "
#                 "Analyze both images in sequence, ensuring each test case covers elements or interactions present in both images, "
#                 "reflecting any form fields, buttons, or UI components that change between images. "
#                 "Include updates, dependencies, or transitions between the images where relevant."
#             )
#         else:
#             question = (
#                 "Write 5 QA 'Test cases' in valid JSON array format for this image. "
#                 "Use these keys for each test case: 'Test Case ID', 'Description of the Test Case', "
#                 "'Steps to Perform the Test Case', 'Expected Result'. "
#                 "Ensure the JSON starts with '[' and ends with ']', and has no extra text."
#             )

#         if st.session_state.joining_option != "select the joining method" or num_images == 1:
#             # Custom spinner
#             with st.spinner("Generating test cases..."):
#                 with st.container():
#                     # Convert image to base64
#                     image_b64 = convert_image_to_base64("join.png")

#                     # Run the model with the user's question
#                     response = run_language_model(question, image_b64)
#                     print("Raw Response:", response)

#                     # Parse JSON test cases
#                     st.session_state.test_cases = parse_test_cases(response)
#                     # Generate HTML table
#                     html_table = generate_table(response)

#                     # Display the table
#                     st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
#                     st.markdown(
#                         f"""
#                         <div style="background-color: #262730; padding: 16px; border-radius: 8px; max-width: 5000px; color: white;">
#                             <h3 style="margin-top: 0;">Response:</h3>
#                             <p style="font-size: 16px;">{html_table}</p>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )

#                     if isinstance(st.session_state.test_cases, list):
#                         # Dropdown for selecting a test case ID
#                         test_case_ids = [tc.get('Test Case ID', f"Unknown {i+1}") for i, tc in enumerate(st.session_state.test_cases)]
#                         selected_id = st.selectbox("Select Test Case ID to Extract:", test_case_ids)

#                         # Button to extract selected test case information
#                         if st.button("Extract Test Case"):
#                             print("extracting test case")
#                             st.session_state.selected_test_case = display_test_case(st.session_state.test_cases, selected_id)
#                             st.write(st.session_state.selected_test_case)

#                         # Button to generate feature file and step definition
#                         # Text input for website URL (placed outside the button click block)
#                         website_url = st.text_input("Enter the website URL (if applicable):", key="website_url")

#                         # Button to generate feature file and step definition
#                         if st.button("Generate Feature File and Step Definition"):

#                             if st.session_state.selected_test_case:
#                                 elements = None

#                                 if website_url:
#                                     with st.spinner("Extracting XPaths from the website..."):
#                                         elements = extract_xpaths(website_url)

#                                         # Print extracted elements with labels
#                                         for element_type, data in elements.items():
#                                             print(f"{element_type.upper()} Elements:")
#                                             for entry in data:
#                                                 print(f"  - Label: {entry['label']}, XPath: {entry['xpath']}")

#                                         driver.quit()
#                                         st.success("XPaths extracted successfully!")

#                                     with st.spinner("Generating feature and step definition files..."):
#                                         # Pass elements instead of xpaths
#                                         feature_file, step_definition = generate_feature_and_step_definition(st.session_state.selected_test_case, elements)
                                        
#                                         if feature_file and step_definition:
#                                             st.subheader("Generated Feature File (Cleaned)")
#                                             st.code(feature_file, language="gherkin")

#                                             st.subheader("Generated Python Step Definition Script (Cleaned)")
#                                             st.code(step_definition, language="python")

#                                         if st.button("Run Behave Tests"):
#                                             st.session_state.run_tests = True
#                             else:
#                                 st.error("No test case selected. Please extract a test case first.")
#                     else:
#                         st.error("Failed to parse test cases. Ensure the response is in JSON format.")
#         else:
#             st.error("Please upload at least one image before pressing 'Done'.")
# -----------------------------------------------------------------------------------------------
# with st.expander("Generate Xpath"):
#        if st.session_state.uploaded_images:
#             num_images = len(st.session_state.uploaded_images)
#             joined_image = None
#             if num_images > 1:
#                 # Join images based on user selection
#                 joined_image = join_images(st.session_state.uploaded_images, st.session_state.joining_option)
#                 joined_image.save("join.png")
#             else:
#                 joined_image = st.session_state.uploaded_images[0]
#                 joined_image.save("join.png")

#             with st.spinner("Generating"):
#                 image_path = "join.png"
#                 image_base64 = image_to_base64(image_path)
#                 client = Groq(api_key='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')
                
#                 print("At the step  1")
#                 response1 = ""
#                 try:
#                     completion = client.chat.completions.create(
#                         model="llama-3.2-90b-vision-preview",

#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     {
#                                         "type": "text",
#                                         "text": "Please give me complete html code for the ui image of form i have given."
#                                         "The html element should cover all the fields present in the ui form present in the image"
#                                         "Make sure all the interactive element present in the ui image of form should be present in the html code also."
#                                         "Dont give the css code. Response shouldn't include any other information other than code."
#                                     },
#                                     {
#                                         "type": "image_url",
#                                         "image_url": {
#                                             "url": f"data:image/jpeg;base64,{image_base64}"
#                                         }
#                                     }
#                                 ]
#                             }
#                         ],
#                         temperature=0,
#                         max_tokens=1024,
#                         top_p=1,
#                         stream=True,
#                         stop=None,
#                     )
#                     # Process and print the response

#                     for chunk in completion:

#                         if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:

#                             print(chunk.choices[0].delta.content, end="")
#                             response1 += str(chunk.choices[0].delta.content)

#                 except Exception as e:

#                     print(f"An error occurred: {e}")

#                 print("code generated-------------------- step 1")
#                 print(response1)

#                 response2 = ""
#                 client = Groq(api_key='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')

#                 completion = client.chat.completions.create(
#                     model="llama-3.2-3b-preview",
#                     messages=[
#                                 {

#                             "role": "user",

#                             "content": "I will give you an html code. Your task is to create an xpath for each interactive elements (input, button, radiobutton, etc) present in the webapge of that html code."
#                             "Xpath should be complete. It should start from html tag"
#                             "xpath should be in this format-- /html/body/div[1]/main/div[5]/div/div/div[2]/section[2]/div[3]/a "
#                             "Output format should be a json which contain key as name of interactive element that is name of value interactive element going to hold and value as the xpath"
#                             "Html code is "+response1
#                         }
#                     ],
#                     temperature=0,
#                     max_tokens=1024,
#                     top_p=1,
#                     stream=True,
#                     stop=None,
#                 )

#                 for chunk in completion:
#                     print(chunk.choices[0].delta.content or "", end="")
#                     response2 += str(chunk.choices[0].delta.content)

#                 start_index = response2.find("json") + len("json")
#                 end_index = response2.find("", start_index)
#                 json_part = str(response2[start_index:end_index])
#                 st.code(json_part, language="json")

#                 print(json_part)

#                 data = json.loads(json_part)
#                 st.download_button(
#                 label="Download Xpath File",
#                 data=json_part,
#                 file_name="xpath.json",
#                 mime="text/json"
#             )

#         # # Write the dictionary to a JSON file
#         # with open('xpath.json', 'w') as json_file:
#         #     json.dump(data, json_file, indent=4)  # indent=4 makes the JSON readable

#         # print("JSON data has been written to 'xpath.json'")
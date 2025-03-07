# Import necessary libraries
from behave import given, then
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the webdriver
@given('I am on the login page')
def step_impl(context):
    context.driver = webdriver.Chrome()
    context.driver.get('https://opensource-demo.orangehrmlive.com/web/index.php/auth/login')

@then('I should see the OrangeHRM logo')
def step_impl(context):
    orangehrm_logo_xpath = '/html/body/div[1]/div[1]/div/div[1]/div/div[1]/img'
    WebDriverWait(context.driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, orangehrm_logo_xpath))
    )
    assert context.driver.find_element(By.XPATH, orangehrm_logo_xpath).is_displayed()

@then('I should see the login form with fields for username and password')
def step_impl(context):
    username_input_xpath = '/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[2]/form[1]/div[1]/div[1]/div[2]/input[1]'
    password_input_xpath = '/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[2]/form[1]/div[2]/div[1]/div[2]/input[1]'
    WebDriverWait(context.driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, username_input_xpath))
    )
    WebDriverWait(context.driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, password_input_xpath))
    )
    assert context.driver.find_element(By.XPATH, username_input_xpath).is_displayed()
    assert context.driver.find_element(By.XPATH, password_input_xpath).is_displayed()

    # Clean up
    context.driver.quit()
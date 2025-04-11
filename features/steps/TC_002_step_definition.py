import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from behave import given, when, then

@given('I am on the login page')
def step_impl(context):
    context.driver = webdriver.Chrome()
    context.driver.get("https://opensource-demo.orangehrmlive.com/web/index.php/auth/login")
    time.sleep(4)

@when('I enter a valid username and password')
def step_impl(context):
    username_input = WebDriverWait(context.driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[2]/form[1]/div[1]/div[1]/div[2]/input[1]"))
    )
    username_input.send_keys("Admin")
    time.sleep(4)
    
    password_input = context.driver.find_element(By.XPATH, "/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[2]/form[1]/div[2]/div[1]/div[2]/input[1]")
    password_input.send_keys("admin123")
    time.sleep(4)

@when('I click the "{button}" button')
def step_impl(context, button):
    login_button = context.driver.find_element(By.XPATH, "/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[2]/div[2]/form[1]/div[3]/button[1]")
    login_button.click()
    time.sleep(4)

@then('I should be redirected to the dashboard page')
def step_impl(context):
    dashboard_title = context.driver.title
    assert dashboard_title == "OrangeHRM"
    time.sleep(4)
    context.driver.quit()
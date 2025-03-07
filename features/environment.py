from selenium import webdriver

def before_scenario(context, scenario):
    # Initialize the WebDriver before each scenario
    context.driver = webdriver.Chrome()  # Make sure chromedriver is installed and in PATH
    context.driver.maximize_window()

def after_scenario(context, scenario):
    # Quit the WebDriver after each scenario
    if hasattr(context, "driver"):
        context.driver.quit()
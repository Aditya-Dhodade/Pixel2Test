Feature: Login Functionality
    Scenario: Valid username and password
        Given I am on the login page
        When I enter a valid username and password
        And I click the "Login" button
        Then I should be redirected to the dashboard page
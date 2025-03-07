Feature: Verify that the login page is displayed correctly
  Scenario: Verify that the login page is displayed correctly
    Given I am on the login page
    Then I should see the OrangeHRM logo
    And I should see the login form with fields for username and password
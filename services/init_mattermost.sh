#!/bin/bash

export MMCTL_INSECURE_TLS_CONNECTIONS=true
export MMCTL_LOCAL=true

until curl -s http://localhost:8065/api/v4/system/ping | grep -q "OK"; do
    printf "Waiting for Mattermost to be fully up..."
    sleep 5
done
echo "Mattermost is up!"

echo "Configuring system to skip setup wizard..."
mmctl --local config set TeamSettings.EnableUserCreation true
mmctl --local config set TeamSettings.EnableOpenServer true
mmctl --local config set ServiceSettings.EnableTutorial false
mmctl --local config set ServiceSettings.EnableOnboardingFlow false
mmctl --local config set ServiceSettings.EnableDesktopLandingPage false

echo "Disabling email notifications..."
mmctl --local config set EmailSettings.SendEmailNotifications false
mmctl --local config set EmailSettings.RequireEmailVerification false
mmctl --local config set EmailSettings.SendPushNotifications false

touch /tmp/mattermost_initialized

echo "Mattermost initialization complete."


import os
import random
import string
import requests
from urllib.parse import urlparse
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

import os

# Slack bot token and channel ID
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
CHANNEL_ID = \
    "CU10HKH1B" # Slovene Office 
    # "C0LDUDHSP" # General
FOLDER_PATH = "./db"  

client = WebClient(token=SLACK_BOT_TOKEN)

def random_filename(extension):
    """Generate a random filename with the specified extension."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)) + extension

def get_file_extension(url):
    """Extract the file extension from the URL."""
    parsed_url = urlparse(url)
    root, extension = os.path.splitext(parsed_url.path)
    return extension if extension else ".jpg"  # Default to .jpg if no extension is found

def save_profile_picture(url, folder_path):
    """Save the profile picture in the specified folder if it's not already present."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        image_data = response.content
        file_size = len(image_data)
        
        # Check if the folder already contains a file of the same size
        for existing_file in os.listdir(folder_path):
            existing_file_path = os.path.join(folder_path, existing_file)
            if os.path.isfile(existing_file_path) and os.path.getsize(existing_file_path) == file_size:
                print(f"File with the same size already exists for {folder_path}. Skipping download.")
                return

        # Get the correct file extension from the URL and save the image
        file_extension = get_file_extension(url)
        file_name = random_filename(file_extension)
        with open(os.path.join(folder_path, file_name), 'wb') as f:
            f.write(image_data)
            print(f"Saved {file_name} in {folder_path}.")

def get_usernames_and_pictures(channel_id):
    try:
        # Get the list of member IDs in the channel
        response = client.conversations_members(channel=channel_id)
        member_ids = response['members']

        for user_id in member_ids:
            user_profile = client.users_profile_get(user=user_id)
            profile = user_profile['profile']

            is_bot = (profile.get("api_app_id") or profile.get("bot_id")) is not None

            # Skip if the user is a bot
            if is_bot:
                print(f"Skipping bot: {profile.get('display_name') or profile.get('real_name')}")
                continue

            username = profile.get("real_name") or profile.get("display_name")
            profile_picture_url = profile.get("image_512")

            # Create folder for each user
            user_folder = os.path.join(FOLDER_PATH, username)
            os.makedirs(user_folder, exist_ok=True)
            
            # Save profile picture in the user's folder
            if profile_picture_url:
                save_profile_picture(profile_picture_url, user_folder)
                
    except SlackApiError as e:
        print(f"Error retrieving data: {e.response['error']}")

# Run the function
get_usernames_and_pictures(CHANNEL_ID)

import discord
import cv2
import numpy as np
from solve_puzzle import solve_queens_puzzle, cv2_imshow
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the token from environment variables
TOKEN = os.getenv('DISCORD_TOKEN')

# Discord Bot Integration
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

# Update the Discord bot message handling to accommodate the new return
@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the message has an attachment
    if message.attachments:
        for attachment in message.attachments:
            # Check if the attachment is an image
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']):
                try:
                    # Download the image
                    image_bytes = await attachment.read()
                    
                    # Convert bytes to numpy array
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Solve the puzzle
                    solution_image, sample_image = solve_queens_puzzle(image_array)
                    
                    if solution_image is not None:
                        # Convert solution image to bytes for Discord
                        solution_bytes = cv2_imshow(solution_image)
                        sample_bytes = cv2_imshow(sample_image)
                        
                        # Send the solution and sample images
                        await message.channel.send(
                            "Here's the solution to the Queens Puzzle!", 
                            file=discord.File(solution_bytes, filename='solution.png')
                        )
                        await message.channel.send(
                            "Here are the sampled pixels for color mapping:", 
                            file=discord.File(sample_bytes, filename='color_map_samples.png')
                        )
                    else:
                        await message.channel.send("Sorry, I couldn't solve the puzzle. Make sure the image is in the correct format.")
                
                except Exception as e:
                    await message.channel.send(f"An error occurred: {e}")

# Use TOKEN when running the bot (from .env file)
client.run(TOKEN)
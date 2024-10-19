import boto3
from email.message import EmailMessage
import os

def send_email(sender, recipient, subject, body_text, attachment_path, region):
    try:
        ses_client = boto3.client('ses', region_name=region)
        
        # Create an email message object
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient
        msg.set_content(body_text)
        
        # Check if the attachment exists
        if os.path.isfile(attachment_path):
            with open(attachment_path, 'rb') as img_file:
                img_data = img_file.read()
                img_name = os.path.basename(attachment_path)
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=img_name)
        else:
            print(f"Attachment file {attachment_path} not found.")
            return
        
        # Send the email via AWS SES
        response = ses_client.send_raw_email(
            Source=sender,
            Destinations=[recipient],
            RawMessage={'Data': msg.as_bytes()}
        )
        print(f"Email sent! Message ID: {response['MessageId']}")
    except Exception as e:
        print(f"Failed to send email: {e}")

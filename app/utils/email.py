# app/utils/email.py
import yagmail
import os
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────
# Email Sending Utility
def send_email_report(recipient_email, subject, body, pdf_path):
    sender_email = os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    yag = yagmail.SMTP(user=sender_email, password=sender_password)
    yag.send(
        to=recipient_email,
        subject=subject,
        contents=[body, pdf_path]
    )

import os
import logging
import smtplib
from email.message import EmailMessage
from threading import Thread
from twilio.rest import Client

log = logging.getLogger(__name__)

# Load config
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_SMS_FROM")       # e.g. "+1234567890"
SMS_TO = os.getenv("SMS_TO")                      # e.g. "+0987654321"

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
RTO_EMAIL = os.getenv("RTO_EMAIL")

def send_sms_alert(message: str):
    """Sends an SMS via Twilio API."""
    if not TWILIO_SID or not TWILIO_AUTH:
        log.warning("Twilio credentials missing. Skipping SMS.")
        return

    def _send():
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH)
            client.messages.create(
                body=message,
                from_=TWILIO_FROM,
                to=SMS_TO
            )
            log.info("SMS alert sent successfully.")
        except Exception as e:
            log.error("Failed to send SMS alert: %s", e)

    Thread(target=_send, daemon=True).start()

def send_rto_email(plate_number: str, camera_id: str, dwell_time: float, image_path: str):
    """Sends a formal violation notice to the RTO via Email with the image attached."""
    if not SMTP_USER or not SMTP_PASS or not RTO_EMAIL:
        log.warning("SMTP credentials missing. Skipping RTO Email.")
        return

    def _send():
        try:
            msg = EmailMessage()
            msg['Subject'] = f"AUTOMATED TICKET: Illegal Parking Violation - {plate_number}"
            msg['From'] = SMTP_USER
            msg['To'] = RTO_EMAIL

            body = (
                f"Official Notice of Traffic Violation\n"
                f"------------------------------------\n"
                f"Vehicle Plate: {plate_number}\n"
                f"Location/Camera: {camera_id}\n"
                f"Dwell Time: {dwell_time} seconds\n\n"
                f"An illegally parked vehicle was detected in a restricted zone. "
                f"Please find the photographic evidence attached to this email.\n\n"
                f"- Smart City Automated Surveillance System"
            )
            msg.set_content(body)

            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img:
                    img_data = img.read()
                    msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)

            log.info("RTO Ticket Email sent successfully for vehicle %s", plate_number)
        except Exception as e:
            log.error("Failed to send RTO Email: %s", e)

    Thread(target=_send, daemon=True).start()
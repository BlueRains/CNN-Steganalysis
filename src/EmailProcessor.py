"""Contains a email processor that can get emails and email attachments."""

import email
import imaplib
import io
from email.policy import default

from PIL import Image
from PIL.ImageFile import ImageFile


class EmailProcessor:
    """Class that creates a continuous connection to a IMAP server.

    You can get the email UUIDs or the image attachements in the email.
    """

    def __init__(
        self, email_address: str, password: str, imap_server: str, imap_port=993
    ):
        """Create a EmailProcessor that connects to `imap_server`.

        Args:
            email_address (str): The email to connect with.
            password (str): The password to use.
            imap_server (str): The IP address to connect to
            imap_port (int, optional): Which port to connect to. Defaults to 993.
        """
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
        self.mail.login(self.email_address, self.password)

    def fetch_emails(self, folder="inbox") -> list[int]:
        """Fetch all emails from the given folder.

        Args:
            folder (str, optional): The folder to search in. Defaults to "inbox".

        Returns:
            list[int]: A list of email UUIDs
        """
        self.mail.select(folder)
        _, data = self.mail.search(None, "ALL")
        return data[0].split()

    def extract_images_from_email(self, email_id: int) -> list[ImageFile]:
        """Extract image from email.

        Args:
            email_id (int): The UUID of the email

        Returns:
            _type_: _description_
        """
        _, data = self.mail.fetch(email_id, "(RFC822)")
        raw_email = data[0][1]
        msg = email.message_from_bytes(raw_email, policy=default)
        images = []
        for part in msg.iter_attachments():
            if part.get_content_maintype() == "image":
                image_data = part.get_payload(decode=True)
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
        return images

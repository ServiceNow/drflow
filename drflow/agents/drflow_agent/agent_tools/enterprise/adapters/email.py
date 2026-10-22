"""Email service adapter implementation for IMAP integration

Expects configuration from drflow enterprise space with structure:
{
    "url": "http://localhost:1143",  # Full URL with dynamically assigned port
    "host_port": 1143,                # Dynamically assigned Docker port
    "name": "email_imap",
    "credentials": {
        "username": "user@example.com",
        "password": "password"
    }
}
"""

import email
import imaplib
import logging
from email.header import decode_header
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from ..base import AuthMethod, BaseServiceAdapter, ServiceCapabilities

logger = logging.getLogger(__name__)


class EmailAdapter(BaseServiceAdapter):
    """Adapter for email service integration via IMAP"""

    def __init__(self, config: Dict[str, Any], session: Optional[requests.Session] = None):
        super().__init__("email_imap", config, session or requests.Session())
        self.auth_method = AuthMethod.BASIC

        url = config.get("url", "")
        if url:
            parsed = urlparse(url)
            self.imap_host = parsed.hostname or "localhost"
        else:
            self.imap_host = "localhost"

        self.imap_port = config.get("host_port", 1143)

        self.credentials = config.get("credentials", {})
        self.imap_connection = None
        self.imap_credentials = None

    def discover_capabilities(self) -> Dict[str, Any]:
        """Discover email service capabilities and endpoints"""
        capabilities = []
        endpoints = {}

        if not self.credentials:
            logger.warning("Email service: No credentials provided in configuration")
            return {
                "capabilities": [],
                "endpoints": {},
                "auth_method": self.auth_method,
                "credentials": {},
                "error": "No credentials provided in configuration",
            }

        username = self.credentials.get("username")
        password = self.credentials.get("password")

        if not username or not password:
            logger.warning("Email service: Invalid credentials in configuration")
            return {
                "capabilities": [],
                "endpoints": {},
                "auth_method": self.auth_method,
                "credentials": {},
                "error": "Invalid credentials in configuration",
            }

        logger.debug(f"Testing email credentials for user: {username}")

        if self._test_imap_access(username, password):
            capabilities.extend(
                [
                    ServiceCapabilities.MESSAGE_SEARCH,
                    ServiceCapabilities.FILE_LISTING,
                    ServiceCapabilities.FILE_DOWNLOAD,
                ]
            )

            endpoints["imap_server"] = f"imap://{self.imap_host}:{self.imap_port}"
            if self.base_url:
                endpoints["web_interface"] = self.base_url

            self.imap_credentials = {"username": username, "password": password}
        else:
            logger.error(f"Email service: Failed to authenticate with provided credentials")
            return {
                "capabilities": [],
                "endpoints": {},
                "auth_method": self.auth_method,
                "credentials": {},
                "error": "Authentication failed with provided credentials",
            }

        self.capabilities = capabilities
        self.endpoints = endpoints

        return {
            "capabilities": capabilities,
            "endpoints": endpoints,
            "auth_method": self.auth_method,
            "credentials": {"username": username},
            "imap_host": self.imap_host,
            "imap_port": self.imap_port,
        }

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Test authentication with provided credentials"""
        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            return False

        if self._test_imap_access(username, password):
            self.credentials = credentials
            self.imap_credentials = credentials
            return True

        return False

    def search(self, terms: List[str], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for emails matching the given terms"""
        results = []

        try:
            if not self._connect_imap():
                logger.error("Failed to connect to IMAP server")
                return results

            status, mailbox_info = self.imap_connection.select("INBOX")
            if status != "OK":
                logger.error(f"Failed to select INBOX: {mailbox_info}")
                return results
            
            logger.debug(f"Selected INBOX, mailbox info: {mailbox_info}")

            search_criteria = self._build_search_criteria(terms)
            logger.debug(f"Search criteria: {search_criteria}")

            status, message_ids = self.imap_connection.search(None, search_criteria)
            logger.debug(f"Search status: {status}, message_ids: {message_ids}")
            
            if status != "OK" or not message_ids[0]:
                logger.debug("Complex search failed, trying simpler fallback")
                if terms:
                    simple_criteria = f'SUBJECT "{terms[0]}"'
                    logger.debug(f"Fallback criteria: {simple_criteria}")
                    status, message_ids = self.imap_connection.search(None, simple_criteria)
                    logger.debug(f"Fallback search status: {status}, message_ids: {message_ids}")

            if status == "OK":
                if message_ids[0]:
                    msg_list = message_ids[0].split()
                    logger.debug(f"Found {len(msg_list)} messages")
                    
                    for msg_id in msg_list[-20:]:
                        try:
                            email_data = self._fetch_email_details(msg_id)
                            if email_data:
                                results.append(email_data)
                        except Exception as e:
                            logger.debug(f"Error fetching email {msg_id}: {e}")
                else:
                    logger.debug("No messages found matching search criteria")
            else:
                logger.error(f"Search failed with status: {status}")

        except Exception as e:
            logger.error(f"Email search failed: {e}")
        finally:
            self._disconnect_imap()

        return results

    def list_files(self, path: str = "/") -> List[Dict[str, Any]]:
        """List emails in a given folder (mailbox)"""
        results = []

        try:
            if not self._connect_imap():
                logger.error("Failed to connect to IMAP server")
                return results

            folder = "INBOX" if path == "/" else path.strip("/")

            status, response = self.imap_connection.select(folder)
            if status != "OK":
                logger.error(f"Failed to select mailbox {folder}")
                return results

            status, message_ids = self.imap_connection.search(None, "ALL")

            if status == "OK":
                for msg_id in message_ids[0].split()[-50:]:
                    try:
                        email_data = self._fetch_email_summary(msg_id)
                        if email_data:
                            results.append(email_data)
                    except Exception as e:
                        logger.debug(f"Error fetching email {msg_id}: {e}")

        except Exception as e:
            logger.error(f"Email listing failed: {e}")
        finally:
            self._disconnect_imap()

        return results

    def download_file(self, file_path: str) -> Dict[str, Any]:
        """Download an email or attachment"""
        parts = file_path.split("/")

        if len(parts) < 2:
            return {"success": False, "error": "Invalid email path"}

        folder = parts[0]
        msg_id = parts[1]

        try:
            if not self._connect_imap():
                return {"success": False, "error": "Failed to connect to IMAP server"}

            self.imap_connection.select(folder)

            status, data = self.imap_connection.fetch(msg_id, "(RFC822)")

            if status == "OK":
                raw_email = data[0][1]
                msg = email.message_from_bytes(raw_email)

                if len(parts) > 2:
                    attachment_name = parts[2]
                    for part in msg.walk():
                        if part.get_content_disposition() == "attachment":
                            filename = part.get_filename()
                            if filename == attachment_name:
                                content = part.get_payload(decode=True)
                                return {
                                    "success": True,
                                    "content": content,
                                    "filename": filename,
                                    "content_type": part.get_content_type(),
                                }
                    return {"success": False, "error": f"Attachment {attachment_name} not found"}
                else:
                    email_content = self._extract_email_content(msg)
                    return {
                        "success": True,
                        "content": email_content.encode("utf-8"),
                        "filename": f"email_{msg_id}.txt",
                        "content_type": "text/plain",
                    }

        except Exception as e:
            logger.error(f"Email download failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self._disconnect_imap()

        return {"success": False, "error": "Failed to download email"}

    def parse_response(self, response: Any, operation: str) -> Any:
        """Parse email service response"""
        return response

    def _test_imap_access(self, username: str, password: str) -> bool:
        """Test IMAP access with credentials"""
        try:
            print(f"Testing IMAP access for {username}")
            imap = imaplib.IMAP4(self.imap_host, self.imap_port)
            imap.login(username, password)
            imap.logout()
            return True
        except Exception as e:
            logger.debug(f"IMAP test failed for {username}: {e}")
            print(f"IMAP test failed for {username}: {e}")
            return False

    def _connect_imap(self) -> bool:
        """Establish IMAP connection"""
        if self.imap_connection:
            return True

        if not self.imap_credentials:
            logger.error("Cannot connect to IMAP: No valid credentials available")
            return False

        try:
            self.imap_connection = imaplib.IMAP4(self.imap_host, self.imap_port)
            username = self.imap_credentials.get("username")
            password = self.imap_credentials.get("password")
            self.imap_connection.login(username, password)
            return True
        except Exception as e:
            logger.error(f"IMAP connection failed: {e}")
            self.imap_connection = None
            return False

    def _disconnect_imap(self):
        """Close IMAP connection"""
        if self.imap_connection:
            try:
                self.imap_connection.close()
                self.imap_connection.logout()
            except:
                pass
            self.imap_connection = None

    def _build_search_criteria(self, terms: List[str]) -> str:
        """Build IMAP search criteria from search terms"""
        if not terms:
            return "ALL"

        if len(terms) == 1:
            term = terms[0]
            return f'OR OR SUBJECT "{term}" FROM "{term}" BODY "{term}"'
        
        all_criteria = []
        for term in terms:
            term_criteria = f'OR OR SUBJECT "{term}" FROM "{term}" BODY "{term}"'
            all_criteria.append(term_criteria)
        
        if len(all_criteria) == 1:
            return all_criteria[0]
        elif len(all_criteria) == 2:
            return f'OR ({all_criteria[0]}) ({all_criteria[1]})'
        else:
            result = all_criteria[0]
            for criteria in all_criteria[1:]:
                result = f'OR ({result}) ({criteria})'
            return result

    def _fetch_email_details(self, msg_id: bytes) -> Optional[Dict[str, Any]]:
        """Fetch detailed information about an email"""
        try:
            status, data = self.imap_connection.fetch(msg_id, "(RFC822.HEADER BODY[TEXT])")

            if status == "OK":
                header_data = data[0][1]
                body_data = data[1][1] if len(data) > 1 else b""

                msg = email.message_from_bytes(header_data)

                subject = self._decode_header(msg.get("Subject", ""))
                from_addr = self._decode_header(msg.get("From", ""))
                to_addr = self._decode_header(msg.get("To", ""))
                date_str = msg.get("Date", "")

                try:
                    body_text = body_data.decode("utf-8", errors="ignore")[:500]
                except:
                    body_text = "Unable to decode body"

                return {
                    "id": msg_id.decode(),
                    "type": "email",
                    "subject": subject,
                    "from": from_addr,
                    "to": to_addr,
                    "date": date_str,
                    "preview": body_text,
                    "path": f"INBOX/{msg_id.decode()}",
                    "has_attachments": self._has_attachments(msg),
                }

        except Exception as e:
            logger.debug(f"Error fetching email details: {e}")
            return None

    def _fetch_email_summary(self, msg_id: bytes) -> Optional[Dict[str, Any]]:
        """Fetch summary information about an email"""
        try:
            status, data = self.imap_connection.fetch(msg_id, "(RFC822.HEADER)")

            if status == "OK":
                header_data = data[0][1]
                msg = email.message_from_bytes(header_data)

                subject = self._decode_header(msg.get("Subject", ""))
                from_addr = self._decode_header(msg.get("From", ""))
                date_str = msg.get("Date", "")

                return {
                    "id": msg_id.decode(),
                    "name": subject,
                    "type": "email",
                    "from": from_addr,
                    "date": date_str,
                    "path": f"INBOX/{msg_id.decode()}",
                    "size": len(header_data),
                }

        except Exception as e:
            logger.debug(f"Error fetching email summary: {e}")
            return None

    def _decode_header(self, header_value: str) -> str:
        """Decode email header value"""
        if not header_value:
            return ""

        decoded_parts = []
        for part, encoding in decode_header(header_value):
            if isinstance(part, bytes):
                try:
                    decoded_parts.append(part.decode(encoding or "utf-8", errors="ignore"))
                except:
                    decoded_parts.append(str(part))
            else:
                decoded_parts.append(str(part))

        return " ".join(decoded_parts)

    def _extract_email_content(self, msg: email.message.Message) -> str:
        """Extract readable content from email message"""
        content_parts = []

        content_parts.append(f"Subject: {self._decode_header(msg.get('Subject', ''))}")
        content_parts.append(f"From: {self._decode_header(msg.get('From', ''))}")
        content_parts.append(f"To: {self._decode_header(msg.get('To', ''))}")
        content_parts.append(f"Date: {msg.get('Date', '')}")
        content_parts.append("\n---\n")

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        content_parts.append(body)
                    except:
                        pass
                elif content_type == "text/html" and len(content_parts) == 5:
                    try:
                        html_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        import re

                        text_body = re.sub("<[^<]+?>", "", html_body)
                        content_parts.append(text_body)
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                content_parts.append(body)
            except:
                content_parts.append("Unable to decode email body")

        attachments = self._list_attachments(msg)
        if attachments:
            content_parts.append("\n---\nAttachments:")
            for att in attachments:
                content_parts.append(f"- {att}")

        return "\n".join(content_parts)

    def _has_attachments(self, msg: email.message.Message) -> bool:
        """Check if email has attachments"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    return True
        return False

    def _list_attachments(self, msg: email.message.Message) -> List[str]:
        """List attachment filenames"""
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)
        return attachments

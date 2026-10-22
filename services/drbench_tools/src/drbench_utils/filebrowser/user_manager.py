"""FileBrowser user management utilities."""

import logging
import os
import subprocess
import time

logger = logging.getLogger("filebrowser.user_manager")


class FileBrowserUserManager:
    """Manages FileBrowser users through environment variables and process restart."""

    def __init__(self):
        """Initialize the FileBrowser user manager."""
        pass

    def update_user(self, username: str, password: str) -> bool:
        """Update FileBrowser user by setting environment variables and restarting the service.
        
        FileBrowser will read these environment variables on startup and create/update the user.
        
        Args:
            username: Username to set
            password: Password to set
            
        Returns:
            True if the service was restarted successfully, False otherwise
        """
        try:
            logger.info(f"Updating FileBrowser user '{username}'...")
            
            os.environ["FILEBROWSER_USER"] = username
            os.environ["FILEBROWSER_PASSWORD"] = password
            
            env_file = "/tmp/filebrowser_env"
            with open(env_file, "w") as f:
                f.write(f"export FILEBROWSER_USER='{username}'\n")
                f.write(f"export FILEBROWSER_PASSWORD='{password}'\n")
            
            logger.info("Environment variables set. Restarting FileBrowser service...")
            
            subprocess.run(["pkill", "-f", "filebrowser.*8090"], check=False)
            
            time.sleep(3)
            
            result = subprocess.run(
                ["pgrep", "-f", "filebrowser.*8090"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"FileBrowser service restarted successfully with user '{username}'")
                return True
            else:
                logger.error("FileBrowser service did not restart properly")
                return False
                
        except Exception as e:
            logger.error(f"Error updating FileBrowser user '{username}': {e}")
            return False

    def create_user(self, username: str, password: str, **_kwargs) -> bool:
        """Create or update a FileBrowser user.
        
        This is an alias for update_user since FileBrowser will create the user
        if it doesn't exist when using environment variables.
        
        Args:
            username: Username to create/update
            password: Password to set
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_user(username, password)
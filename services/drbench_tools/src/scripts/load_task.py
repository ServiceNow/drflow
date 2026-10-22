import json
import logging
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from drbench_utils.mattermost import MattermostUserManager
from drbench_utils.nextcloud import NextcloudUserManager

logger = logging.getLogger(__name__)

TASK_DIR = Path("/drbench/task")
TASK_FILE = TASK_DIR / "task.json"
ENV_FILE = TASK_DIR / "env.json"

NEXTCLOUD_APP = "nextcloud"
MATTERMOST_APP = "mattermost"
EMAIL_APP = "email"
FILE_SYSTEM_APP = "file_system"
FILEBROWSER_APP = "filebrowser"

NEXTCLOUD_APP_ADMIN = "admin"
NEXTCLOUD_APP_PASSWORD = "admin_pwd"
MATTERMOST_APP_ADMIN = "mm_admin"
MATTERMOST_APP_PASSWORD = "mm_admin_pwd"
EMAIL_APP_ADMIN = "admin"
EMAIL_APP_PASSWORD = "admin_pwd"
MATTERMOST_SAFE_TEAM_PREFIX = "team-"
MATTERMOST_RESTRICTED_TEAM_PREFIXES = (
    "admin",
    "api",
    "channel",
    "claim",
    "error",
    "files",
    "help",
    "landing",
    "login",
    "mfa",
    "oauth",
    "plug",
    "plugins",
    "playbooks",
    "post",
    "signup",
)

SUPPORTED_APPS = [NEXTCLOUD_APP, MATTERMOST_APP, EMAIL_APP, FILE_SYSTEM_APP]


def load_task():
    logger.info("Loading task...")

    with open(ENV_FILE, "r") as f:
        task_env_data = json.load(f)

    env_files_data = task_env_data.get("env_files", [])

    if not all("app" in evf for evf in env_files_data):
        logger.error("One or more env files do not have an 'app' key. Exiting...")
        return False

    if not all(evf.get("app") in SUPPORTED_APPS for evf in env_files_data):
        unsupported_apps = set([evf.get("app") for evf in env_files_data if evf.get("app") not in SUPPORTED_APPS])
        logger.error(
            "One or more env files have an unsupported app ({}). Exiting...".format(", ".join(sorted(unsupported_apps)))
        )
        return False

    app_credentials = task_env_data.get("app_credentials", {})
    credentials_override = {
        ac["app"]: {"username": ac["username"], "password": ac["password"]} for ac in app_credentials
    }

    process_nextcloud_files(
        [evf for evf in env_files_data if evf.get("app") == NEXTCLOUD_APP],
        credentials=credentials_override.get(NEXTCLOUD_APP, None),
    )

    process_mattermost_files(
        [evf for evf in env_files_data if evf.get("app") == MATTERMOST_APP],
        credentials=credentials_override.get(MATTERMOST_APP, None),
    )

    process_email_files(
        [evf for evf in env_files_data if evf.get("app") == EMAIL_APP],
        credentials=credentials_override.get(EMAIL_APP, None),
    )

    process_file_system_files(
        [evf for evf in env_files_data if evf.get("app") in [FILE_SYSTEM_APP, FILEBROWSER_APP]],
        credentials=credentials_override.get(FILEBROWSER_APP, credentials_override.get(FILE_SYSTEM_APP, None)),
    )

    logger.info("Task loaded successfully!")
    return True




def process_nextcloud_files(env_files_data, credentials=None):
    """Process Nextcloud environment files."""
    if credentials:
        username = credentials.get("username", NEXTCLOUD_APP_ADMIN)
        password = credentials.get("password", "")
        nc_manager = NextcloudUserManager()
        nc_manager.create_user(username, password)
    else:
        username = NEXTCLOUD_APP_ADMIN
        password = "admin_pwd"

    for env_file_data in env_files_data:
        source = env_file_data.get("source", "")
        file_type = env_file_data.get("type", "file")

        if file_type == "users" or source.endswith("users.csv"):
            if not process_nextcloud_users(env_file_data):
                logger.error(f"Failed to process Nextcloud user data: {source}")
                return False
        else:
            if not copy_nextcloud_file(env_file_data, user_folder=username):
                logger.error(f"Failed to process Nextcloud env file: {source}")
                return False

    subprocess.run(["php", "occ", "files:scan", "--all"], check=True, cwd="/var/www/nextcloud")
    logger.info("Nextcloud files re-scanned successfully!")
    return True


def copy_nextcloud_file(env_file_data, user_folder=None):
    """
    Copy the env file to the Nextcloud data directory.

    The env file is copied to the user's files directory in Nextcloud.
    If the destination is a directory, the env file is copied with its original name.
    If the destination is a file, the env file is copied with the specified name.
    If the destination is not specified, the env file is copied to the user's files directory with its original name.

    After copying, the source env file is removed.

    Args:
       env_file_data: A dictionary containing the env file data.

    Returns:
       bool: True if the env file was copied successfully, False otherwise.
    """
    source = env_file_data.get("source")
    env_file_path = TASK_DIR / source
    if not env_file_path.exists():
        logger.error(f'Env file "{source}" does not exist. Exiting...')
        return False
    try:

        username = user_folder or NEXTCLOUD_APP_ADMIN
        destination = env_file_data.get("destination", "")
        dest_path = Path(f"/var/www/nextcloud/data/{username}/files/{destination.lstrip('/')}")
        if dest_path.is_dir():
            dest_path = dest_path / env_file_path.name

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_file_path, dest_path)
        env_file_path.unlink()
        logger.info(f"Env file {source} copied to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to copy env file {source} to {dest_path}. Error: {e}")
        return False
    return True




def run_mmctl_command(command_args: list, error_message: str, check_output: bool = True):
    """
    Helper function to run mmctl commands and handle common errors.

    Args:
        command_args (list): List of arguments for subprocess.run.
        error_message (str): Custom error message to display on failure.
        check_output (bool): If True, raise CalledProcessError on non-zero exit code.

    Returns:
        str: The standard output of the command.

    Raises:
        FileNotFoundError: If mmctl is not found.
        subprocess.CalledProcessError: If the mmctl command fails.
        json.JSONDecodeError: If JSON output cannot be parsed.
    """
    try:
        result = subprocess.run(["mmctl", "--local"] + command_args, capture_output=True, text=True, check=check_output)
        return result.stdout
    except FileNotFoundError:
        print("Error: 'mmctl' command not found. Ensure it's in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: {error_message} Command: {' '.join(['mmctl', '--local'] + command_args)}")
        print(f"  Exit Code: {e.returncode}")
        print(f"  Stdout: {e.stdout.strip()}")
        print(f"  Stderr: {e.stderr.strip()}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON output for command: {' '.join(['mmctl', '--local'] + command_args)}")
        raise


def add_user_to_all_teams_and_channels(user_username: str):
    """
    Adds a specified user to all existing Mattermost teams and then
    makes them join every open (public) channel within those teams
    using mmctl --local.

    Args:
        user_username (str): The username of the user to add.
    """
    print(f"Starting process for user: '{user_username}'")
    try:
        print("\nFetching all team names...")
        teams_json_output = run_mmctl_command(["team", "list", "--format", "json"], "Failed to list teams.")
        if teams_json_output is None:
            print("No teams found or mmctl command failed. Exiting.")
            return False
        teams = json.loads(teams_json_output)
        team_names = [team["name"] for team in teams]

        if not team_names:
            print("No teams found. Exiting.")
            return

        print(f"Found {len(team_names)} teams: {', '.join(team_names)}")

        for team_name in team_names:
            print(f"\n--- Processing Team: '{team_name}' ---")

            print(f"Attempting to add user '{user_username}' to team '{team_name}'...")
            try:
                res = run_mmctl_command(
                    ["team", "users", "add", team_name, user_username],
                    f"Failed to add user '{user_username}' to team '{team_name}'.",
                    check_output=False,
                )
                if res is None:
                    print(f"Failed to add user '{user_username}' to team '{team_name}'. Exiting.")
                    continue
                print(f"Successfully added '{user_username}' to '{team_name}'.")
            except subprocess.CalledProcessError as e:
                if "already a member" in e.stderr:
                    print(f"User '{user_username}' is already a member of '{team_name}'. Skipping team add.")
                else:
                    print(f"ERROR: Failed to add user '{user_username}' to team '{team_name}': {e.stderr.strip()}")
                    continue
            except Exception as e:
                print(f"ERROR: An unexpected error occurred while processing team '{team_name}': {e}")
                continue

            print(f"Fetching public channels for team '{team_name}'...")
            channels_json_output = run_mmctl_command(
                ["channel", "list", team_name, "--format", "json"],
                f"Failed to list public channels for team '{team_name}'.",
            )
            if channels_json_output is None:
                print(f"Failed to list channels for team '{team_name}'. Skipping.")
                continue
            channels = json.loads(channels_json_output)
            open_channels = [ch["name"] for ch in channels if ch.get("type") == "O"]

            if not open_channels:
                print(f"No open channels found in team '{team_name}'.")
                continue

            print(f"Found {len(open_channels)} open channels in '{team_name}': {', '.join(open_channels)}")

            for channel_name in open_channels:
                print(
                    f"  Attempting to add user '{user_username}' to channel '{channel_name}' in team '{team_name}'..."
                )
                try:
                    res = run_mmctl_command(
                        ["channel", "users", "add", team_name + ":" + channel_name, user_username],
                        f"Failed to add user '{user_username}' to channel '{channel_name}' in team '{team_name}'.",
                        check_output=False,
                    )
                    if res is None:
                        print(f"ERROR: Failed to add user '{user_username}' to channel '{channel_name}'. Skipping.")
                        continue
                    print(f"  Successfully added '{user_username}' to channel '{channel_name}'.")
                except subprocess.CalledProcessError as e:
                    if "already a member" in e.stderr:
                        print(
                            f"  User '{user_username}' is already a member of channel '{channel_name}'. Skipping channel add."
                        )
                    else:
                        print(f"  Error adding user '{user_username}' to channel '{channel_name}': {e.stderr.strip()}")
                except Exception as e:
                    print(f"  An unexpected error occurred while processing channel '{channel_name}': {e}")

        print(f"\n--- Process completed for user '{user_username}'. ---")

    except Exception as e:
        print(f"\nAn overall error occurred: {e}")
        return False

    return True


def process_mattermost_files(env_files_data, credentials=None):
    has_errors = False
    for env_file_data in env_files_data:
        source = env_file_data.get("source", "")
        file_type = env_file_data.get("type", "file")

        if file_type == "users" or source.endswith("users.csv"):
            if not process_mattermost_users(env_file_data):
                logger.error(f"Failed to process Mattermost user data: {source}")
                has_errors = True
                continue
        else:
            if not process_mattermost_data(env_file_data):
                logger.error(f"ERROR: LOAD TASK: Failed to process Mattermost env file: {source}")
                has_errors = True
                continue

    if credentials:
        username = credentials.get("username", MATTERMOST_APP_ADMIN)
        password = credentials.get("password", MATTERMOST_APP_PASSWORD)
        mm_manager = MattermostUserManager()
        if not mm_manager.update_user(username=username, password=password):
            logger.error(f"Failed to update/create Mattermost user '{username}'")
            print(f"M A TT ER MO S T : Failed to update/create Mattermost user '{username}'")
        else:
            print(f"M A TT ER MO S T : Successfully updated/created Mattermost user '{username}'")
    else:
        username = MATTERMOST_APP_ADMIN
        password = MATTERMOST_APP_PASSWORD
        mm_manager = MattermostUserManager()
        if not mm_manager.update_user(username=username, password=password):
            logger.error(f"Failed to update/create default Mattermost user '{username}'")

    # Add user to all teams if he is not already a member to ensure he has access to all channels and conversations.
    if not add_user_to_all_teams_and_channels(username):
        logger.error(f"Failed to add user '{username}' to all Mattermost teams.")
        has_errors = True

    if not has_errors:
        logger.info("Mattermost files processed successfully!")
    else:
        logger.error("Some Mattermost files could not be processed successfully. Check the logs for details.")
    return not has_errors


def process_mattermost_data(env_file_data):
    """
    Process the env file for Mattermost.

    The env file is zipped and then imported using mmctl import.

    Args:
        env_file_data: A dictionary containing the env file data.

    Returns:
        bool: True if the env file was processed successfully, False otherwise.
    """
    source = env_file_data.get("source")
    env_file_path = TASK_DIR / source
    if not env_file_path.exists():
        logger.error(f'Env file "{source}" does not exist. Exiting...')
        return False
    try:
        normalization_stats = normalize_mattermost_jsonl_schema(env_file_path)
        if normalization_stats["timestamps"]:
            logger.info(
                "Normalized %s Mattermost post timestamps from 'created_at' to 'create_at' in %s",
                normalization_stats["timestamps"],
                source,
            )
        if normalization_stats["user_roles"]:
            logger.info(
                "Normalized %s Mattermost user role field(s) to the string schema expected by bulk import in %s",
                normalization_stats["user_roles"],
                source,
            )
        if normalization_stats["team_names"]:
            logger.info(
                "Normalized %s Mattermost team name(s) with reserved prefixes in %s",
                normalization_stats["team_names"],
                source,
            )

        zip_file = env_file_path.with_suffix(".zip")

        with zipfile.ZipFile(zip_file, "w") as zf:
            zf.write(env_file_path, arcname=env_file_path.name)

        logger.info(f"Env file {source} zipped to {zip_file}")
        logger.info(f"Launching import job for env file {source}...")
        output = run_mmctl_command(
            ["import", "process", "--json", "--bypass-upload", str(zip_file)],
            f"Failed to launch import job for env file {source}.",
        )
        if output is None:
            logger.error(f"Failed to launch import job for env file {source}. Exiting...")
            return False
        output = json.loads(output)[0]
        job_id = output.get("id")
        t0 = time.time()
        while time.time() - t0 < 60:
            logger.info(f"Waiting for import job {job_id} to finish...")
            time.sleep(1)
            output = run_mmctl_command(
                ["import", "job", "show", job_id, "--json"],
                f"Failed to show import job {job_id}.",
            )
            if output is None:
                logger.error(f"Failed to show import job {job_id}. Exiting...")
                return False
            output = json.loads(output)[0]
            status = output.get("status")
            if status == "success":
                logger.info(f"Import job {job_id} completed successfully!")
                break
            elif status == "error":
                error = output.get("data").get("error")
                logger.error(f"Import job {job_id} failed with error: {error}")
                return False
            else:
                logger.info(f"Import job {job_id} is in {status} state...")
        else:
            logger.error(f"Import job {job_id} timed out after 60 seconds.")
            return False

        zip_file.unlink()
        env_file_path.unlink()
        logger.info(f"Mattermost file {source} processed successfully!")
    except Exception as e:
        logger.error(f"Failed to process env file {source}. Error: {e}")
        return False
    return True


def normalize_mattermost_jsonl_schema(jsonl_path: Path) -> dict[str, int]:
    """Normalize common Mattermost bulk import schema mismatches before mmctl import."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    normalization_stats = {"timestamps": 0, "user_roles": 0, "team_names": 0}
    normalization_stats["team_names"] = normalize_mattermost_team_names(items)

    for item in items:
        item_type = item.get("type")
        payload_key = "post" if item_type == "post" else "direct_post" if item_type == "direct_post" else None
        if item_type == "user":
            user = item.get("user", {})
            roles = user.get("roles")
            if isinstance(roles, list):
                user["roles"] = normalize_mattermost_user_roles(roles)
                normalization_stats["user_roles"] += 1

        if payload_key:
            payload = item.get(payload_key, {})
            if "created_at" in payload and "create_at" not in payload:
                payload["create_at"] = payload.pop("created_at")
                normalization_stats["timestamps"] += 1

    if any(normalization_stats.values()):
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=True))
                f.write("\n")

    return normalization_stats


def normalize_mattermost_user_roles(roles) -> str:
    """Map legacy role arrays to Mattermost's expected space-delimited role string."""
    if not isinstance(roles, list):
        return roles

    role_map = {
        "member": "system_user",
        "user": "system_user",
        "admin": "system_admin",
    }
    normalized_roles = []
    for role in roles:
        if not isinstance(role, str):
            continue
        mapped_role = role_map.get(role, role)
        if mapped_role not in normalized_roles:
            normalized_roles.append(mapped_role)

    if not normalized_roles:
        return "system_user"

    if "system_admin" in normalized_roles and "system_user" not in normalized_roles:
        normalized_roles.insert(0, "system_user")

    return " ".join(normalized_roles)


def needs_reserved_team_prefix_fix(name: str) -> bool:
    lower_name = name.lower()
    return any(lower_name.startswith(prefix) for prefix in MATTERMOST_RESTRICTED_TEAM_PREFIXES)


def make_unique_mattermost_name(candidate: str, taken_names: set[str]) -> str:
    if candidate not in taken_names:
        return candidate

    suffix = 1
    while f"{candidate}-{suffix}" in taken_names:
        suffix += 1
    return f"{candidate}-{suffix}"


def sanitize_mattermost_team_name(name: str, taken_names: set[str]) -> str:
    sanitized = (name or "").strip().lower()
    sanitized = re.sub(r"[^a-z0-9_-]", "-", sanitized)
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")

    if not sanitized or not sanitized[0].isalpha():
        sanitized = f"{MATTERMOST_SAFE_TEAM_PREFIX}{sanitized}".strip("-")

    if needs_reserved_team_prefix_fix(sanitized):
        sanitized = f"{MATTERMOST_SAFE_TEAM_PREFIX}{sanitized}"

    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-")
    if not sanitized:
        sanitized = "team-space"

    unique_name = make_unique_mattermost_name(sanitized, taken_names)
    taken_names.add(unique_name)
    return unique_name


def normalize_mattermost_team_names(items) -> int:
    team_name_map = {}
    taken_names = set()

    for item in items:
        if item.get("type") != "team":
            continue
        team = item.get("team", {})
        original_name = team.get("name")
        if not original_name:
            continue

        sanitized_name = sanitize_mattermost_team_name(original_name, taken_names)
        if sanitized_name != original_name:
            team["name"] = sanitized_name
            team_name_map[original_name] = sanitized_name

    if not team_name_map:
        return 0

    for item in items:
        item_type = item.get("type")
        if item_type == "channel":
            channel = item.get("channel", {})
            team_name = channel.get("team")
            if team_name in team_name_map:
                channel["team"] = team_name_map[team_name]
        elif item_type == "user":
            user = item.get("user", {})
            for team in user.get("teams", []) or []:
                team_name = team.get("name")
                if team_name in team_name_map:
                    team["name"] = team_name_map[team_name]
        elif item_type in {"post", "direct_post"}:
            payload_key = "post" if item_type == "post" else "direct_post"
            payload = item.get(payload_key, {})
            team_name = payload.get("team")
            if team_name in team_name_map:
                payload["team"] = team_name_map[team_name]

    return len(team_name_map)




def process_email_files(env_files_data, credentials=None):
    """Process email environment files."""
    logger.info("Processing email environment files...")

    for env_file_data in env_files_data:
        if not process_email_data(env_file_data, credentials=credentials):
            logger.error(f"Failed to process email env file: {env_file_data.get('source')}")
            return False

    if credentials:
        from drbench_utils.email import EmailManager

        username = credentials.get("username", EMAIL_APP_ADMIN)
        password = credentials.get("password", EMAIL_APP_PASSWORD)
        email = credentials.get("email", f"{username}@drbench.com")

        logger.info(f"Updating email user '{username}'...")
        email_manager = EmailManager()

        if not email_manager.update_user_password(username, password, email):
            logger.error(f"Failed to update password for email user '{username}'")
            return False

        logger.info(f"Successfully updated email user '{username}'")

    logger.info("Email files processed successfully!")
    return True


def is_self_contained_email_jsonl(jsonl_path):
    """Check if a JSONL file contains self-contained email data (users + emails).

    A file is considered self-contained if it contains either:
    1. Explicit user definitions (entries with "type": "user")
    2. Email entries that would create implicit users

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        bool: True if the file appears to be self-contained
    """
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if data.get("type") == "user":
                        return True
                    if "from" in data and ("to" in data or "cc" in data):
                        return True
                except:
                    continue

        return False
    except:
        return False


def process_email_data(env_file_data, credentials=None):
    """
    Process the env file for email system.

    The env file is expected to be a self-contained JSONL file with both users and emails.

    Args:
        env_file_data: A dictionary containing the env file data.

    Returns:
        bool: True if the env file was processed successfully, False otherwise.
    """
    source = env_file_data.get("source")
    env_file_path = TASK_DIR / source

    if not env_file_path.exists():
        logger.error(f'Email env file "{source}" does not exist. Exiting...')
        return False

    file_suffix = env_file_path.suffix.lower()

    try:
        if file_suffix in [".jsonl", ".json"]:
            logger.info(f"Loading self-contained email data from {env_file_path}...")

            from drbench_utils.email import load_email_data

            success = load_email_data(env_file_path)

            if success:
                logger.info("Email data initialization completed successfully!")
                env_file_path.unlink()
                logger.info(f"Email env file {source} processed and removed")
                return True
            else:
                logger.error("Email data initialization failed")
                return False

        else:
            logger.error(f"Unsupported email file type: {file_suffix}")
            logger.error("Expected a self-contained JSONL file with both users and emails")
            return False

    except Exception as e:
        logger.error(f"Failed to process email env file {source}. Error: {e}")
        return False




def process_file_system_files(env_files_data, credentials=None):
    """Process file system environment files."""

    logger.info("Processing file system environment files...")

    if credentials:
        username = credentials.get("username", "admin")
        password = credentials.get("password", "admin_pwd")
        logger.info(f"Updating FileBrowser user '{username}'...")
        
        from drbench_utils.filebrowser import FileBrowserUserManager
        fb_manager = FileBrowserUserManager()
        
        if not fb_manager.update_user(username, password):
            logger.error(f"Failed to update FileBrowser user '{username}'")
    else:
        logger.info("FileBrowser will use default authentication")

    for env_file_data in env_files_data:
        if not copy_file_system_file(env_file_data, credentials=credentials):
            logger.error(f"Failed to process file system env file: {env_file_data.get('source')}")
            return False

    logger.info("File system files processed successfully!")
    return True


def copy_file_system_file(env_file_data, credentials=None):
    """
    Copy the env file to the user's home directory for FileBrowser access.

    The env file is copied to the user's home directory based on the VNC_USER environment variable.
    If the destination is a directory, the env file is copied with its original name.
    If the destination is a file, the env file is copied with the specified name.
    If the destination is not specified, the env file is copied to the user's home directory with its original name.

    After copying, the source env file is removed.

    Args:
       env_file_data: A dictionary containing the env file data.

    Returns:
       bool: True if the env file was copied successfully, False otherwise.
    """
    source = env_file_data.get("source")
    env_file_path = TASK_DIR / source
    if not env_file_path.exists():
        logger.error(f'Env file "{source}" does not exist. Exiting...')
        return False

    try:
        user = os.environ.get("VNC_USER", "root")

        if user == "root":
            user_home = Path("/root")
        else:
            user_home = Path(f"/home/{user}")

        destination = env_file_data.get("destination", "")
        if destination:
            dest_path = user_home / destination
        else:
            dest_path = user_home / env_file_path.name

        if dest_path.is_dir():
            dest_path = dest_path / env_file_path.name

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(env_file_path, dest_path)

        env_file_path.unlink()

        logger.info(f"File system env file {source} copied to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to copy file system env file {source} to {dest_path}. Error: {e}")
        return False

    return True




def process_mattermost_users(env_file_data):
    """
    Process user data file for Mattermost.

    Args:
        env_file_data: A dictionary containing the env file data with 'source' pointing to a CSV file.

    Returns:
        bool: True if users were created successfully, False otherwise.
    """
    source = env_file_data.get("source")
    env_file_path = TASK_DIR / source

    if not env_file_path.exists():
        logger.error(f'User data file "{source}" does not exist.')
        return False

    try:
        logger.info(f"Creating Mattermost users from {source}...")

        from drbench_utils.mattermost import MattermostUserManager

        manager = MattermostUserManager()

        success = manager.create_all_users(str(env_file_path))

        if success:
            logger.info("Mattermost users created successfully!")
            env_file_path.unlink()
            logger.info(f"User data file {source} processed and removed")
            return True
        else:
            logger.error("Failed to create Mattermost users")
            return False

    except Exception as e:
        logger.error(f"Failed to process Mattermost user data file {source}. Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def process_nextcloud_users(env_file_data):
    """
    Process user data file for Nextcloud.

    Args:
        env_file_data: A dictionary containing the env file data with 'source' pointing to a CSV file.

    Returns:
        bool: True if users were created successfully, False otherwise.
    """
    source = env_file_data.get("source")
    env_file_path = TASK_DIR / source

    if not env_file_path.exists():
        logger.error(f'User data file "{source}" does not exist.')
        return False

    try:
        logger.info(f"Creating Nextcloud users from {source}...")

        from drbench_utils.nextcloud import NextcloudUserManager

        manager = NextcloudUserManager()

        success = manager.create_all_users(str(env_file_path))

        if success:
            logger.info("Nextcloud users created successfully!")
            env_file_path.unlink()
            logger.info(f"User data file {source} processed and removed")
            return True
        else:
            logger.error("Failed to create Nextcloud users")
            return False

    except Exception as e:
        logger.error(f"Failed to process Nextcloud user data file {source}. Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    if not TASK_DIR.exists():
        logger.error(f"{TASK_DIR} does not exist. Exiting...")
        return False
    if not TASK_DIR.is_dir():
        logger.error(f"{TASK_DIR} is not a directory. Exiting...")
        return False
    logger.info("Starting task loader...")
    if not TASK_FILE.exists():
        logger.warning(f"{TASK_FILE} does not exist. Exiting...")
        return False

    load_task_ok = load_task()

    if load_task_ok:
        logger.info("Task loaded successfully!")
        return True

    logger.error("Failed to load task. Exiting...")
    return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()

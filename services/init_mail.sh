#!/bin/bash

echo "Configuring mail system..."

EMAIL_DOMAIN=${EMAIL_DOMAIN:-"drbench.com"}
DBENCH_USER=${DBENCH_USER:-"root"}

VMAIL_USER=${VMAIL_USER:-vmail}

VMAIL_UID=$(id -u ${VMAIL_USER})
VMAIL_GID=$(id -g ${VMAIL_USER})


postconf -e "myhostname = $(hostname)"
postconf -e "mydestination = localhost.localdomain, localhost, $(hostname)"
postconf -e "mynetworks = 127.0.0.0/8 [::ffff:127.0.0.0]/104 [::1]/128"
postconf -e "inet_interfaces = all"
postconf -e "home_mailbox = Maildir/"

postconf -e "master_service_disable = smtp"
sed -i 's/^smtp      inet/#smtp      inet/' /etc/postfix/master.cf
echo "1025      inet  n       -       y       -       -       smtpd" >> /etc/postfix/master.cf

postconf -e "virtual_mailbox_domains = ${EMAIL_DOMAIN}, company.com, external.com"
postconf -e "virtual_mailbox_base = /var/mail"
postconf -e "virtual_mailbox_maps = hash:/etc/postfix/virtual_mailboxes"
postconf -e "virtual_minimum_uid = $(id -u ${VMAIL_USER})"
postconf -e "virtual_uid_maps = static:$(id -u ${VMAIL_USER})"
postconf -e "virtual_gid_maps = static:$(id -g ${VMAIL_USER})"

postconf -e "smtpd_relay_restrictions = permit_mynetworks permit_sasl_authenticated defer_unauth_destination"
postconf -e "smtpd_recipient_restrictions = permit_mynetworks permit_sasl_authenticated reject_unauth_destination"

# Disable external relay to keep emails internal only
postconf -e "relayhost ="
postconf -e "relay_domains ="

touch /etc/postfix/virtual_mailboxes
chown ${DBENCH_USER}:${DBENCH_USER} /etc/postfix/virtual_mailboxes
chmod 644 /etc/postfix/virtual_mailboxes

cat > /etc/dovecot/dovecot.conf << EOF
protocols = imap
listen = *
log_path = /dev/stdout
mail_location = mbox:/var/mail/%u:INBOX=/var/mail/%u/inbox

service imap-login {
  inet_listener imap {
    port = 1143
  }
}

passdb {
  driver = passwd-file
  args = /etc/dovecot/passwd
}

userdb {
  driver = passwd-file
  args = /etc/dovecot/passwd
  default_fields = home=/var/mail/%u uid=${VMAIL_USER} gid=${VMAIL_USER}
}

disable_plaintext_auth = no
auth_mechanisms = plain login

mail_privileged_group = ${VMAIL_USER}
dotlock_use_excl = no
mbox_write_locks = fcntl

namespace inbox {
  inbox = yes
  location = 
  mailbox Drafts {
    special_use = \Drafts
  }
  mailbox Junk {
    special_use = \Junk
  }
  mailbox Sent {
    special_use = \Sent
  }
  mailbox "Sent Messages" {
    special_use = \Sent
  }
  mailbox Trash {
    special_use = \Trash
  }
}
EOF

if ! id ${VMAIL_USER} &>/dev/null; then
  useradd -r ${VMAIL_USER} -d /var/mail -s /bin/false
fi

echo "Initializing Dovecot password file..."
> /etc/dovecot/passwd

if [ -f /usr/share/roundcube/config/config.inc.php ]; then
    echo "Configuring Roundcube..."
    
    cat > /usr/share/roundcube/config/config.inc.php << EOF
<?php
\$config = array();
\$config['db_dsnw'] = 'sqlite:////var/lib/roundcube/roundcube.db?mode=0646';
\$config['default_host'] = 'localhost:1143';
\$config['smtp_server'] = 'localhost:1025';
\$config['smtp_auth_type'] = null;
\$config['smtp_user'] = '';
\$config['smtp_pass'] = '';
\$config['des_key'] = 'changeme1234567890abcdef';
\$config['plugins'] = array('archive', 'zipdownload');
\$config['skin'] = 'elastic';
\$config['debug_level'] = 4;
\$config['log_dir'] = '/tmp/';

\$config['drafts_mbox'] = 'Drafts';
\$config['junk_mbox'] = 'Junk';
\$config['sent_mbox'] = 'sent';
\$config['trash_mbox'] = 'Trash';

\$config['create_default_folders'] = true;

\$config['imap_auto_subscribe'] = true;

\$config['use_subscriptions'] = true;
?>
EOF

    cd /usr/share/roundcube
    php -q ./bin/initdb.sh --dir=./SQL || echo "Database may already be initialized"
fi

echo "Stopping any existing mail services..."
pkill -f postfix || true
pkill -f dovecot || true
sleep 1

touch /tmp/mail_initialized

echo "Mail system configuration complete!"
echo "Email users will be loaded from CSV data by init_mail_data.py"

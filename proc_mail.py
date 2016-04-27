# -*- coding: utf-8 -*-

"""
This file contains functions of processing emails
"""

import re
from html2text import html2text


def get_mail_body(_mail_file):
    """
    Extract the mail body from the mail file.
    :param mail_file: file contains the raw content of email
    :type mail_file: str
    :return: string that contains the body of an email
    :rtype: str
    :param _mail_file:
    :type _mail_file:
    :return:
    :rtype:
    """
    with open(_mail_file, 'r', encoding='utf-8', errors='ignore') as f:
        msg = f.read()
    body_start = msg.find('\n\n') + len('\n\n')

    return msg[body_start:]


def get_mail_text(_msg_body):
    """
    Get plain text and remove html tags in the messge body.
    :param _msg_body:
    :type _msg_body:
    :return: the string that does not contain html tags.
    :rtype: str
    """
    if '<html' in _msg_body:
        return html2text(_msg_body)
    else:
        return _msg_body


def sub_entities(_plain_msg):
    """
    sub-stitute the entities: url, email address, number, dollar.
    :param _plain_msg: plain text message
    :type _plain_msg: str
    :return: plain text without unwanted entities
    :rtype: str
    """
    _sub_url = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' urladdr ',
                      _plain_msg, flags=re.MULTILINE)
    _sub_eml = re.sub(r'[\w\.-]+@[\w\.-]+', ' mailaddr ', _sub_url, flags=re.MULTILINE)
    _sub_num = re.sub(r'\b\d+\b', ' number ', _sub_eml, flags=re.MULTILINE)
    _sub_dol = _sub_num.replace('$', ' dollar ')
    _sub_usc = _sub_dol.replace('_', ' ')

    return _sub_usc


def process_mail(_mail_file):
    """
    wrap the processing functions.
    :param _mail_file:
    :type _mail_file:
    :return:
    :rtype:
    """
    return sub_entities(get_mail_text(get_mail_body(_mail_file)))

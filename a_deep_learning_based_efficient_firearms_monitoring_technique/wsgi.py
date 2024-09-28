"""
WSGI config for a_deep_learning_based_efficient_firearms_monitoring_technique

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'a_deep_learning_based_efficient_firearms_monitoring_technique.settings')
application = get_wsgi_application()

"""AWS SigV4 signing for Bedrock Mantle."""
from __future__ import annotations

import os

import httpx


def _sign_bedrock(url: str, body: bytes, route: dict) -> dict:
    """Return a headers dict with AWS SigV4 Authorization for a Bedrock POST."""
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    # Profile comes from AWS_PROFILE env var (set in the user's shell before starting the router).
    # Not stored in router_config.json so the config is shareable without exposing profile names.
    profile = os.environ.get("AWS_PROFILE")
    region = route.get("aws_region", "us-east-1")

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    creds = session.get_credentials().get_frozen_credentials()
    req = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(creds, "bedrock", region).add_auth(req)
    return dict(req.headers)


class _SigV4Auth(httpx.Auth):
    """Apply AWS SigV4 signing to every request the openai SDK makes.

    httpx calls auth_flow synchronously before each send. At that point the
    openai SDK has already serialised the body to bytes, so request.content is
    available for body-hash computation.
    """

    def __init__(self, route: dict) -> None:
        self._route = route

    def auth_flow(self, request: httpx.Request):
        signed = _sign_bedrock(str(request.url), request.content, self._route)
        for k, v in signed.items():
            request.headers[k.lower()] = v
        yield request

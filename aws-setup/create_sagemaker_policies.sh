#!/bin/bash

# 1. Create trust policy
cat > sagemaker-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# 2. Create S3 policy
cat > sagemaker-s3-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:ListBucketVersions"
      ],
      "Resource": [
        "arn:aws:s3:::chinese-produce-forecast-346761359662",
        "arn:aws:s3:::chinese-produce-forecast-346761359662/*"
      ]
    }
  ]
}
EOF

echo "✅ Policy files created"

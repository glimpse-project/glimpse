Appart from the .travis.yml at the top of the repo these scripts help support
our Travis-CI-based continuous integration.

`build-travis-docker-image.sh` builds a Docker image with most of our
build-time prerequisites which helps speed up repeat builds of Glimpse.

To use, this you should review and edit the `Dockerfile` that's here and run:

```
./build-travis-docker-image.sh
docker login
docker push rib1/glimpse-travis
```

*(Note: the image we're using currently is named rib1/glimpse-travis, but since
that implies being able to login as rib1 you might need to replace references
to that name within these scripts and .travis.yml to use an alternative name)*
*(Note: Dodkerfile expects there to be an adjacent windows-sdk/ and absolute
paths for the -cross-files under windows-sdk/meson/ should assume the SDK will
be copied to /windows-sdk in the docker image)*

`travis-ci-prep-docker-ubuntu.sh` derives a final docker image which knows
about the travis user so that it's possible to mount the travis build directory
into our running Docker image with compatible permissions.

`travis-ci-build.sh` encapsulates all our build instructions so we don't have
to describe everything with the more awkward .travis.yml syntax.


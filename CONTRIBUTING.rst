.. highlight:: shell

============
Contributing
============

Your contributions are much appreciated! Feel free to contribute in one of these ways:


Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/YosefLab/Cassiopeia/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Ideally, when you are fixing a bug, please first provide a test that breaks
due to the bug. Your contributed code should then fix this test.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Cassiopeia could always use more documentation, whether as part of the
official Cassiopeia docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/YosefLab/Cassiopeia/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome

Get Started!
------------

Ready to contribute? Here's how to set up `cassiopeia` for local development.

1. Fork the `cassiopeia` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/Cassiopeia.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv cassiopeia
    $ cd cassiopeia/
    $ make install

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 cassiopeia tests
    $ make test

   To get flake8, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Coding Standards
-----------------------
1. Don't duplicate code. Certainly no blocks longer than a couple of lines. It's almost always better to refactor than to duplicate blocks of code.
2. Almost all code should at least be run by a unit tests. No pull request should decrease unit test coverage by much.
3. Document each new method and each new class with a docstring.
4. Don't commit commented-out code. Just delete it or store it somewhere outside of the repo. You probably aren't going to need it. At worse, it's stored in previous commits, from before it was commented out.
5. A pull request (PR) will typically close at least one Github issue. For these pull requests, write the issue it closes in the description, e.g. ``closes #210``. The issue will be automatically closed when the PR is merged.
6. Don't commit data to the repository, except perhaps a few small (< 50 KB) files of test data.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python >= 3.8. Check
   https://travis-ci.org/YosefLab/Cassiopeia/pull_requests
   and make sure that the tests pass for all supported Python versions.

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.

Also, make sure you've tested your code using tox by running::

$ tox

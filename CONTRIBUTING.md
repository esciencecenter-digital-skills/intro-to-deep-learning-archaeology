# Contributing

This lesson is part of The Carpentries Incubator.
[The Carpentries][c-site] ([Software Carpentry][swc-site], [Data Carpentry][dc-site],
and [Library Carpentry][lc-site]) are open source projects,
and we welcome contributions of all kinds:
new lessons,
fixes to existing material,
bug reports,
and reviews of proposed changes are all welcome.

## Contributor Agreement

By contributing,
you agree that we may redistribute your work under [our license](LICENSE.md).
In exchange,
we will address your issues and/or assess your change proposal as promptly as we can,
and help you become a member of our community.
Everyone involved in [The Carpentries][c-site]
agrees to abide by our [code of conduct](CODE_OF_CONDUCT.md).

## How to Contribute

The easiest way to get started is to file an issue
to tell us about a spelling mistake,
some awkward wording,
or a factual error.
This is a good way to introduce yourself
and to meet some of our community members.

1.  If you do not have a [GitHub][github] account,
    you can [send us comments by email][email].
    However,
    we will be able to respond more quickly if you use one of the other methods described below.

2.  If you have a [GitHub][github] account,
    or are willing to [create one][github-join],
    but do not know how to use Git,
    you can report problems or suggest improvements by [creating an issue][issues].
    This allows us to assign the item to someone
    and to respond to it in a threaded discussion.

3.  If you are comfortable with Git,
    and would like to add or change material,
    you can submit a pull request (PR).
    Instructions for doing this are [included below](#using-github).

## Where to Contribute

If you wish to change this lesson,
please work in <https://github.com/carpentries-incubator/deep-learning-intro>,
which can be viewed at <https://carpentries-incubator.github.io/deep-learning-intro/>.

## What to Contribute

There are many ways to contribute,
from writing new exercises and improving existing ones
to updating or filling in the documentation
and submitting [bug reports][issues]
about things that don't work, aren't clear, or are missing.
If you are looking for ideas, please see the 'Issues' tab for
a list of issues associated with this repository,
or you may also look at the issues for [Data Carpentry][dc-issues],
[Software Carpentry][swc-issues], and [Library Carpentry][lc-issues] projects.

Comments on issues and reviews of pull requests are just as welcome:
we are smarter together than we are on our own.
Reviews from novices and newcomers are particularly valuable:
it's easy for people who have been using these lessons for a while
to forget how impenetrable some of this material can be,
so fresh eyes are always welcome.

## Pull Requests
If you decide to contribute in the form of a pull request please read the
following carefully.

### Getting the code and making changes
This lesson requires you to have python 3.4+ and ruby 2.7+ installed.
See the [lesson setup](https://carpentries.github.io/lesson-example/setup.html)
for more information.

1. Fork the repository to create a place on github where you can push your changes to.
2. Clone the repository to your local pc.
3. Make your changes
4. Preview your changes using `make serve`
5. If you are happy with your changes check if the site still adheres to the
   lesson format using `make lesson-check-all`
6. Commit and push your changes
7. Create a (draft) pull request

### Working on a draft pull request
When working on changes to this lesson it is useful to create a draft pull
request early in the development.
This alerts the maintainers that you are working on a certain issue and allows
them to provide you with early feedback.

### Submitting a pull request
If you submit a pull request please request one of the maintainers as one of the reviewers.
Each pull request needs to be reviewed by at least one person that is not an author
of the pull request.

Each pull request should also be free of conflicts with the gh-pages branch and build
succesfully with the github action.
The github action to build will be automatically run when you submit a pull request.
The easiest way to check if your contribution is ready is to run:
```bash
$ make site
$ make lesson-check-all
```
These should result in no errors.

Make sure your name and email adress are in the AUTHORS file in your pull request.
Once the pull request is approved it will be merged by one of the maintainers.

### Merge conflicts
Sometimes your contributions can give rise to [merge conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/about-merge-conflicts)
The author of the PR is responsible to resolve merge conflicts.
In case, the author is incapable to do so, a short discussion should be conducted on who is responsible.

## What *Not* to Contribute

Our lessons already contain more material than we can cover in a typical workshop,
so we are usually *not* looking for more concepts or tools to add to them.
As a rule,
if you want to introduce a new idea,
you must (a) estimate how long it will take to teach
and (b) explain what you would take out to make room for it.
The first encourages contributors to be honest about requirements;
the second, to think hard about priorities.

We are also not looking for exercises or other material that only run on one platform.
Our workshops typically contain a mixture of Windows, macOS, and Linux users;
in order to be usable,
our lessons must run equally well on all three.

## Using GitHub

If you choose to contribute via GitHub, you may want to look at
[How to Contribute to an Open Source Project on GitHub][how-contribute].
To manage changes, we follow [GitHub flow][github-flow].
Each lesson has two maintainers who review issues and pull requests or encourage others to do so.
The maintainers are community volunteers and have final say over what gets merged into the lesson.
To use the web interface for contributing to a lesson:

1.  Fork the originating repository to your GitHub profile.
2.  Within your version of the forked repository, move to the `gh-pages` branch and
create a new branch for each significant change being made.
3.  Navigate to the file(s) you wish to change within the new branches and make revisions
    as required.
4.  Commit all changed files within the appropriate branches.
5.  Create individual pull requests from each of your changed branches
to the `gh-pages` branch within the originating repository.
6.  If you receive feedback, make changes using your issue-specific branches of the forked
repository and the pull requests will update automatically.
7.  Repeat as needed until all feedback has been addressed.

When starting work, please make sure your clone of the originating `gh-pages` branch is up-to-date
before creating your own revision-specific branch(es) from there.
Additionally, please only work from your newly-created branch(es) and *not*
your clone of the originating `gh-pages` branch.
Lastly, published copies of all the lessons are available in the `gh-pages` branch of the
originating repository for reference while revising.

## Other Resources

General discussion of [Software Carpentry][swc-site] and [Data Carpentry][dc-site]
happens on the [discussion mailing list][discuss-list],
which everyone is welcome to join.
You can also [reach us by email][email].

[email]: mailto:admin@software-carpentry.org
[dc-issues]: https://github.com/issues?q=user%3Adatacarpentry
[dc-lessons]: http://datacarpentry.org/lessons/
[dc-site]: http://datacarpentry.org/
[discuss-list]: http://lists.software-carpentry.org/listinfo/discuss
[github]: https://github.com
[github-flow]: https://guides.github.com/introduction/flow/
[github-join]: https://github.com/join
[how-contribute]: https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github
[issues]: https://guides.github.com/features/issues/
[swc-issues]: https://github.com/issues?q=user%3Aswcarpentry
[swc-lessons]: https://software-carpentry.org/lessons/
[swc-site]: https://software-carpentry.org/
[c-site]: https://carpentries.org/
[lc-site]: https://librarycarpentry.org/
[lc-issues]: https://github.com/issues?q=user%3Alibrarycarpentry

# Conventions

We below list conventions that we kindly ask all contributors to follow when helping with the content, making fixes or adding/editing lessons. Please consider these hints while you are preparing your pull-requests.

## Terminology

We consistently like to use:
- Deep Learning (both capitalized)

### Validation set vs. test set
Not everybody agrees on the terminology of test set versus validation set. We are sticking to the 
definition that is consistent with the Keras API. According to their definition, a validation set
can be used during training for hyperparameter optimization. Test set should always be kept separate
to do the final evaluation of the model.

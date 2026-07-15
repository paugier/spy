project(
  'my-package', 'c', 'cython',
  version: '0.1.0',
  license: 'MIT',
  meson_version: '>= 1.0.0',
  default_options: [
    'buildtype=release',
    'c_std=c11',
  ]
)

py = import('python').find_installation()

subdir('my_package')

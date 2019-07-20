import rospy, tf2_ros, argparse
import numpy as np
remove_first_slash = lambda s: s if s[0] != '/' else s[1:]


class TFExtractor:
  def __init__(self, from_frame_id, to_frame_id, filename, verbose):
    self.tf_buffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tf_buffer)
    self.from_frame_id = remove_first_slash(from_frame_id)
    self.to_frame_id = remove_first_slash(to_frame_id)
    self.filename = filename
    self.verbose = verbose
    self.transforms = []

  def run(self):
    rate = rospy.Rate(100.0)
    counter = 0
    while not rospy.is_shutdown():
      try:
        t = self.tf_buffer.lookup_transform(self.from_frame_id, self.to_frame_id, rospy.Time())
      except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rate.sleep()
        continue

      stamp = t.header.stamp.to_sec()

      tran = t.transform.translation
      tran = (tran.x, tran.y, tran.z)
      rot = t.transform.rotation
      rot = (rot.w, rot.x, rot.y, rot.z)

      sample = (stamp,) + tran + rot
      self.transforms += [sample]
      counter += 1

      if self.verbose and counter % 100 == 0:
        print '{}: {}'.format(counter, ','.join(map(str, sample)))
      rate.sleep()

  def save(self):
    np.save(self.filename, np.array(self.transforms))
    print 'Finished!'


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Extract TF transformation from a frame to abnother frame.')
  parser.add_argument('-f', '--from-frame-id', required=True, type=str, help='from-frame-id frame id')
  parser.add_argument('-t', '--to-frame-id', required=True, type=str, help='to frame id')
  parser.add_argument('-o', '--output', required=True, type=str, help='File to save the transform array to')
  parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
  args = parser.parse_args()
  rospy.init_node('tf_extractor')
  tfe = TFExtractor(args.from_frame_id, args.to_frame_id, args.output, args.verbose)
  tfe.run()
  tfe.save()
